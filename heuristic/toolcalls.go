package heuristic

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/datar-psa/goeval/api"
)

// ToolArgsMode selects how an expected call's arguments are compared with an actual call's
type ToolArgsMode int

const (
	// ToolArgsSubset (the default) requires every key of the expected arguments to be present
	// in the actual arguments with an equal value; extra actual keys are allowed.
	// Expected calls without arguments check nothing.
	ToolArgsSubset ToolArgsMode = iota
	// ToolArgsExact requires the actual arguments to equal the expected ones (key order ignored)
	ToolArgsExact
	// ToolArgsIgnore compares tool names only
	ToolArgsIgnore
)

// ToolCallsOptions configures the ToolCalls scorer.
//
// The expected calls come from ScoreInputs.ExpectedToolCalls (per case) and Required (all cases).
// With no expected calls, no Required, no Forbidden, no MaxCalls and NoToolCalls unset
// there is nothing to check and the scorer returns ErrNoExpectedValue.
type ToolCallsOptions struct {
	// Required lists tool names that must be called at least once, whatever their arguments.
	// It is always order-free; Ordered applies to ScoreInputs.ExpectedToolCalls only.
	Required []string
	// Forbidden lists tool names that must never be called. One call to any of them scores 0.
	Forbidden []string
	// NoToolCalls requires the trace to be empty (e.g. refusal cases). Any call scores 0.
	NoToolCalls bool
	// MaxCalls caps the number of calls in the trace; 0 means no cap. Exceeding it scores 0.
	MaxCalls int
	// Ordered requires ExpectedToolCalls to appear in the trace in the given order
	// (other calls may be interleaved). Otherwise they may appear in any order.
	Ordered bool
	// Args selects the argument comparison (default ToolArgsSubset)
	Args ToolArgsMode
	// ArgsMatcher, when set, replaces Args: it receives the decoded expected and actual
	// argument objects and reports whether the actual call satisfies the expected one.
	// It is only called when the names are equal and the expected call has arguments.
	ArgsMatcher func(expected, actual map[string]any) bool
}

// ToolCalls returns a scorer that checks the model's tool calls (ScoreInputs.ToolCalls) against expectations.
//
// A forbidden call, a call when NoToolCalls is set, or more than MaxCalls calls score 0.
// Otherwise the score is the fraction of expected calls (ExpectedToolCalls and Required) that were found.
// Metadata lists every miss and violation in "misses" and "violations".
func ToolCalls(opts ToolCallsOptions) api.Scorer {
	return &toolCallsScorer{opts: opts}
}

type toolCallsScorer struct {
	opts ToolCallsOptions
}

func (s *toolCallsScorer) Score(ctx context.Context, in api.ScoreInputs) api.Score {
	result := api.Score{
		Name:     "ToolCalls",
		Metadata: make(map[string]any),
	}

	o := s.opts
	nothingToCheck := len(in.ExpectedToolCalls) == 0 && len(o.Required) == 0 && len(o.Forbidden) == 0 &&
		!o.NoToolCalls && o.MaxCalls <= 0
	if nothingToCheck {
		result.Error = api.ErrNoExpectedValue
		return result
	}

	calls := in.ToolCalls
	violations := []string{}
	misses := []string{}

	if o.NoToolCalls && len(calls) > 0 {
		violations = append(violations, fmt.Sprintf("expected no tool calls, got %d", len(calls)))
	}
	if o.MaxCalls > 0 && len(calls) > o.MaxCalls {
		violations = append(violations, fmt.Sprintf("%d tool calls exceed the maximum of %d", len(calls), o.MaxCalls))
	}
	for i, c := range calls {
		if slices.Contains(o.Forbidden, c.Name) {
			violations = append(violations, fmt.Sprintf("call %d: forbidden tool %q was called", i, c.Name))
		}
	}

	var matched int
	if o.Ordered {
		matched = s.matchOrdered(in.ExpectedToolCalls, calls, &misses)
	} else {
		matched = s.matchUnordered(in.ExpectedToolCalls, calls, &misses)
	}
	for _, name := range o.Required {
		if slices.ContainsFunc(calls, func(c api.ToolCall) bool { return c.Name == name }) {
			matched++
		} else {
			misses = append(misses, fmt.Sprintf("required tool %q was never called", name))
		}
	}

	total := len(in.ExpectedToolCalls) + len(o.Required)
	switch {
	case len(violations) > 0:
		result.Score = 0
	case total == 0:
		result.Score = 1
	default:
		result.Score = float64(matched) / float64(total)
	}

	called := make([]string, len(calls))
	for i, c := range calls {
		called[i] = c.Name
	}
	result.Metadata["expected_count"] = total
	result.Metadata["matched_count"] = matched
	result.Metadata["call_count"] = len(calls)
	result.Metadata["called"] = called
	result.Metadata["misses"] = misses
	result.Metadata["violations"] = violations
	result.Metadata["ordered"] = o.Ordered

	return result
}

// matchOrdered matches expected calls, in order, to later and later calls of the trace.
// Taking the earliest match for each expected call leaves the most room for the rest;
// an expected call with no match is a miss and the search carries on with the next.
func (s *toolCallsScorer) matchOrdered(expected, calls []api.ToolCall, misses *[]string) int {
	matched := 0
	cursor := 0
	for i, e := range expected {
		found := -1
		for j := cursor; j < len(calls); j++ {
			if ok, _ := s.callMatches(e, calls[j]); ok {
				found = j
				break
			}
		}
		if found >= 0 {
			matched++
			cursor = found + 1
			continue
		}
		reason := s.whyMissing(e, calls)
		if slices.ContainsFunc(calls[:cursor], func(c api.ToolCall) bool {
			ok, _ := s.callMatches(e, c)
			return ok
		}) {
			reason = "found only before an earlier expected call (out of order)"
		}
		*misses = append(*misses, fmt.Sprintf("expected call %d (%s): %s", i, e.Name, reason))
	}
	return matched
}

// matchUnordered finds a maximum one-to-one matching between expected and actual calls,
// so an actual call satisfies at most one expected call.
func (s *toolCallsScorer) matchUnordered(expected, calls []api.ToolCall, misses *[]string) int {
	owner := make([]int, len(calls)) // actual call index -> expected index, or -1
	for i := range owner {
		owner[i] = -1
	}
	var try func(e int, seen []bool) bool
	try = func(e int, seen []bool) bool {
		for j := range calls {
			if seen[j] {
				continue
			}
			if ok, _ := s.callMatches(expected[e], calls[j]); !ok {
				continue
			}
			seen[j] = true
			if owner[j] < 0 || try(owner[j], seen) {
				owner[j] = e
				return true
			}
		}
		return false
	}

	matchedExpected := make([]bool, len(expected))
	matched := 0
	for e := range expected {
		if try(e, make([]bool, len(calls))) {
			matched++
		}
	}
	for _, e := range owner {
		if e >= 0 {
			matchedExpected[e] = true
		}
	}
	for i, e := range expected {
		if !matchedExpected[i] {
			*misses = append(*misses, fmt.Sprintf("expected call %d (%s): %s", i, e.Name, s.whyMissing(e, calls)))
		}
	}
	return matched
}

// whyMissing explains why no call satisfies e
func (s *toolCallsScorer) whyMissing(e api.ToolCall, calls []api.ToolCall) string {
	var reasons []string
	for j, c := range calls {
		if c.Name != e.Name {
			continue
		}
		_, why := s.callMatches(e, c)
		reasons = append(reasons, fmt.Sprintf("call %d: %s", j, why))
	}
	if len(reasons) == 0 {
		return "tool was never called"
	}
	if len(reasons) == 1 {
		return "arguments differ (" + reasons[0] + ")"
	}
	return "no call with matching arguments (" + strings.Join(reasons, "; ") + ")"
}

// callMatches reports whether actual satisfies expected, and if not, why
func (s *toolCallsScorer) callMatches(expected, actual api.ToolCall) (bool, string) {
	if expected.Name != actual.Name {
		return false, fmt.Sprintf("name %q != %q", actual.Name, expected.Name)
	}
	if s.opts.ArgsMatcher == nil && s.opts.Args == ToolArgsIgnore {
		return true, ""
	}
	if len(expected.Arguments) == 0 {
		return true, ""
	}
	want, err := expected.DecodeArguments()
	if err != nil {
		return false, fmt.Sprintf("expected arguments are not a JSON object: %v", err)
	}
	if len(want) == 0 && s.opts.Args != ToolArgsExact && s.opts.ArgsMatcher == nil {
		return true, ""
	}
	got, err := actual.DecodeArguments()
	if err != nil {
		return false, fmt.Sprintf("actual arguments are not a JSON object: %v", err)
	}

	if s.opts.ArgsMatcher != nil {
		if s.opts.ArgsMatcher(want, got) {
			return true, ""
		}
		return false, "ArgsMatcher rejected the arguments"
	}
	if s.opts.Args == ToolArgsExact {
		if reflect.DeepEqual(want, got) {
			return true, ""
		}
		return false, fmt.Sprintf("arguments %s != expected %s", argText(actual), argText(expected))
	}
	// ToolArgsSubset
	for k, wv := range want {
		gv, ok := got[k]
		if !ok {
			return false, fmt.Sprintf("missing argument %q", k)
		}
		if !reflect.DeepEqual(wv, gv) {
			return false, fmt.Sprintf("argument %q is %v, expected %v", k, gv, wv)
		}
	}
	return true, ""
}

func argText(c api.ToolCall) string {
	if len(c.Arguments) == 0 {
		return "{}"
	}
	return string(c.Arguments)
}
