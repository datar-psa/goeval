package api

import (
	"encoding/json"
	"testing"
)

func TestToolCallDecodeArguments(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		want    map[string]any
		wantErr bool
	}{
		{name: "empty", args: "", want: map[string]any{}},
		{name: "null", args: "null", want: map[string]any{}},
		{name: "object", args: `{"a":1,"b":"x"}`, want: map[string]any{"a": float64(1), "b": "x"}},
		{name: "not an object", args: `[1]`, wantErr: true},
		{name: "invalid", args: `{`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ToolCall{Arguments: json.RawMessage(tt.args)}.DecodeArguments()
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if len(got) != len(tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
			for k, v := range tt.want {
				if got[k] != v {
					t.Errorf("got[%q] = %v, want %v", k, got[k], v)
				}
			}
		})
	}
}

// Keyed literals written against v0.1.0 must keep compiling and leave the new fields empty.
func TestScoreInputsBackwardCompatible(t *testing.T) {
	in := ScoreInputs{Output: "o", Expected: "e", Input: "i"}
	if in.ToolCalls != nil || in.ExpectedToolCalls != nil {
		t.Error("new fields must default to nil")
	}
}
