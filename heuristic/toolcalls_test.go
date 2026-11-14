package heuristic

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/datar-psa/goeval/api"
)

func call(name, args string) api.ToolCall {
	c := api.ToolCall{Name: name}
	if args != "" {
		c.Arguments = json.RawMessage(args)
	}
	return c
}

func TestToolCalls(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name           string
		opts           ToolCallsOptions
		calls          []api.ToolCall
		expected       []api.ToolCall
		wantErr        error
		wantScore      float64
		wantMiss       string // substring of a miss, if any
		wantViolations int
	}{
		{
			name:      "expected tool called",
			calls:     []api.ToolCall{call("get_holdings", `{"account":"main"}`)},
			expected:  []api.ToolCall{call("get_holdings", "")},
			wantScore: 1,
		},
		{
			name:      "expected tool never called",
			calls:     []api.ToolCall{call("other", "")},
			expected:  []api.ToolCall{call("get_holdings", "")},
			wantScore: 0,
			wantMiss:  "never called",
		},
		{
			name:      "half of the expected calls found",
			calls:     []api.ToolCall{call("a", "")},
			expected:  []api.ToolCall{call("a", ""), call("b", "")},
			wantScore: 0.5,
			wantMiss:  "(b)",
		},
		{
			name:      "no calls at all",
			expected:  []api.ToolCall{call("a", "")},
			wantScore: 0,
			wantMiss:  "never called",
		},
		{
			name:      "unordered accepts any order",
			calls:     []api.ToolCall{call("b", ""), call("a", "")},
			expected:  []api.ToolCall{call("a", ""), call("b", "")},
			wantScore: 1,
		},
		{
			name:      "ordered accepts interleaved calls",
			opts:      ToolCallsOptions{Ordered: true},
			calls:     []api.ToolCall{call("a", ""), call("x", ""), call("b", "")},
			expected:  []api.ToolCall{call("a", ""), call("b", "")},
			wantScore: 1,
		},
		{
			name:      "ordered flags wrong order",
			opts:      ToolCallsOptions{Ordered: true},
			calls:     []api.ToolCall{call("b", ""), call("a", "")},
			expected:  []api.ToolCall{call("a", ""), call("b", "")},
			wantScore: 0.5,
			wantMiss:  "out of order",
		},
		{
			name:      "same tool twice needs two calls",
			calls:     []api.ToolCall{call("a", `{"x":1}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`), call("a", `{"x":1}`)},
			wantScore: 0.5,
		},
		{
			name:      "one call cannot satisfy two expectations but matching reassigns",
			calls:     []api.ToolCall{call("a", `{"x":1,"y":2}`), call("a", `{"x":1}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`), call("a", `{"y":2}`)},
			wantScore: 1,
		},
		{
			name:      "subset ignores extra keys",
			calls:     []api.ToolCall{call("a", `{"x":1,"y":"z"}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 1,
		},
		{
			name:      "subset flags wrong value",
			calls:     []api.ToolCall{call("a", `{"x":2}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 0,
			wantMiss:  `argument "x" is 2, expected 1`,
		},
		{
			name:      "subset flags missing key",
			calls:     []api.ToolCall{call("a", `{}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 0,
			wantMiss:  `missing argument "x"`,
		},
		{
			name:      "subset compares nested values structurally",
			calls:     []api.ToolCall{call("a", `{"f":{"k":[1,2]},"z":0}`)},
			expected:  []api.ToolCall{call("a", `{"f":{"k":[1,2]}}`)},
			wantScore: 1,
		},
		{
			name:      "exact rejects extra keys",
			opts:      ToolCallsOptions{Args: ToolArgsExact},
			calls:     []api.ToolCall{call("a", `{"x":1,"y":2}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 0,
			wantMiss:  "!= expected",
		},
		{
			name:      "exact ignores key order",
			opts:      ToolCallsOptions{Args: ToolArgsExact},
			calls:     []api.ToolCall{call("a", `{"y":2,"x":1}`)},
			expected:  []api.ToolCall{call("a", `{"x":1,"y":2}`)},
			wantScore: 1,
		},
		{
			name:      "ignore compares names only",
			opts:      ToolCallsOptions{Args: ToolArgsIgnore},
			calls:     []api.ToolCall{call("a", `{"x":9}`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 1,
		},
		{
			name: "custom matcher decides",
			opts: ToolCallsOptions{ArgsMatcher: func(want, got map[string]any) bool {
				return strings.EqualFold(want["q"].(string), got["q"].(string))
			}},
			calls:     []api.ToolCall{call("search", `{"q":"ETF"}`)},
			expected:  []api.ToolCall{call("search", `{"q":"etf"}`)},
			wantScore: 1,
		},
		{
			name:      "custom matcher rejects",
			opts:      ToolCallsOptions{ArgsMatcher: func(want, got map[string]any) bool { return false }},
			calls:     []api.ToolCall{call("search", `{"q":"a"}`)},
			expected:  []api.ToolCall{call("search", `{"q":"a"}`)},
			wantScore: 0,
			wantMiss:  "ArgsMatcher",
		},
		{
			name:      "invalid actual arguments do not match",
			calls:     []api.ToolCall{call("a", `{oops`)},
			expected:  []api.ToolCall{call("a", `{"x":1}`)},
			wantScore: 0,
			wantMiss:  "not a JSON object",
		},
		{
			name:      "required tool called",
			opts:      ToolCallsOptions{Required: []string{"a"}},
			calls:     []api.ToolCall{call("a", `{"any":1}`)},
			wantScore: 1,
		},
		{
			name:      "required tool missing",
			opts:      ToolCallsOptions{Required: []string{"a", "b"}},
			calls:     []api.ToolCall{call("a", "")},
			wantScore: 0.5,
			wantMiss:  `required tool "b"`,
		},
		{
			name:           "forbidden call scores zero despite expected calls found",
			opts:           ToolCallsOptions{Forbidden: []string{"delete"}},
			calls:          []api.ToolCall{call("a", ""), call("delete", "")},
			expected:       []api.ToolCall{call("a", "")},
			wantScore:      0,
			wantViolations: 1,
		},
		{
			name:      "forbidden never called",
			opts:      ToolCallsOptions{Forbidden: []string{"delete"}},
			calls:     []api.ToolCall{call("a", "")},
			wantScore: 1,
		},
		{
			name:           "max calls exceeded",
			opts:           ToolCallsOptions{MaxCalls: 2},
			calls:          []api.ToolCall{call("a", ""), call("a", ""), call("a", "")},
			expected:       []api.ToolCall{call("a", "")},
			wantScore:      0,
			wantViolations: 1,
		},
		{
			name:      "max calls respected",
			opts:      ToolCallsOptions{MaxCalls: 2},
			calls:     []api.ToolCall{call("a", ""), call("a", "")},
			expected:  []api.ToolCall{call("a", "")},
			wantScore: 1,
		},
		{
			name:      "no tool calls expected and none made",
			opts:      ToolCallsOptions{NoToolCalls: true},
			wantScore: 1,
		},
		{
			name:           "no tool calls expected but one made",
			opts:           ToolCallsOptions{NoToolCalls: true},
			calls:          []api.ToolCall{call("a", "")},
			wantScore:      0,
			wantViolations: 1,
		},
		{
			name:    "nothing to check",
			wantErr: api.ErrNoExpectedValue,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := ToolCalls(tt.opts).Score(ctx, api.ScoreInputs{ToolCalls: tt.calls, ExpectedToolCalls: tt.expected})

			if res.Name != "ToolCalls" {
				t.Errorf("Name = %q", res.Name)
			}
			if !errors.Is(res.Error, tt.wantErr) {
				t.Fatalf("Error = %v, want %v", res.Error, tt.wantErr)
			}
			if res.Score != tt.wantScore {
				t.Errorf("Score = %v, want %v (metadata %v)", res.Score, tt.wantScore, res.Metadata)
			}
			if tt.wantErr != nil {
				return
			}
			misses := res.Metadata["misses"].([]string)
			if tt.wantMiss != "" && !strings.Contains(strings.Join(misses, "\n"), tt.wantMiss) {
				t.Errorf("misses %q lack %q", misses, tt.wantMiss)
			}
			if tt.wantMiss == "" && res.Score == 1 && len(misses) != 0 {
				t.Errorf("unexpected misses %q", misses)
			}
			if v := res.Metadata["violations"].([]string); len(v) != tt.wantViolations {
				t.Errorf("violations = %q, want %d", v, tt.wantViolations)
			}
		})
	}
}

func TestToolCallsMetadata(t *testing.T) {
	res := ToolCalls(ToolCallsOptions{Ordered: true}).Score(context.Background(), api.ScoreInputs{
		ToolCalls:         []api.ToolCall{call("a", ""), call("b", "")},
		ExpectedToolCalls: []api.ToolCall{call("a", "")},
	})
	if res.Metadata["call_count"] != 2 || res.Metadata["expected_count"] != 1 || res.Metadata["matched_count"] != 1 {
		t.Errorf("metadata = %v", res.Metadata)
	}
	if got := res.Metadata["called"].([]string); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("called = %v", got)
	}
}
