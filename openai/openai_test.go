package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/datar-psa/goeval/api"
)

const testKey = "test-key-not-real-123"

var testSchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"choice": map[string]interface{}{"type": "string"},
	},
	"required":             []string{"choice"},
	"additionalProperties": false,
}

type captured struct {
	path   string
	header http.Header
	body   map[string]any
}

// newServer answers every request with status and body, and records the last one
func newServer(t *testing.T, status int, body string) (*httptest.Server, *captured) {
	t.Helper()
	got := &captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		got.path = r.URL.Path
		got.header = r.Header.Clone()
		_ = json.Unmarshal(raw, &got.body)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

func chatBody(content string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"content": content}, "finish_reason": "stop"}},
	})
	return string(b)
}

func TestStructuredGenerateRequestShape(t *testing.T) {
	srv, got := newServer(t, 200, chatBody(`{"choice":"A"}`))
	g := NewGenerator("openai/gpt-test",
		WithBaseURL(srv.URL+"/api/v1/"),
		WithAPIKey(testKey),
		WithHeader("X-Title", "goeval-test"),
		WithHeader("Authorization", "Bearer attacker"),
		WithExtraBody(map[string]any{"provider": map[string]any{"require_parameters": true}, "model": "ignored"}),
	)

	res, err := g.StructuredGenerate(context.Background(), "hello", testSchema)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res["choice"] != "A" {
		t.Errorf("result = %v, want choice A", res)
	}
	if got.path != "/api/v1/chat/completions" {
		t.Errorf("path = %q", got.path)
	}
	if h := got.header.Get("Authorization"); h != "Bearer "+testKey {
		t.Errorf("Authorization = %q", h)
	}
	if h := got.header.Get("Content-Type"); h != "application/json" {
		t.Errorf("Content-Type = %q", h)
	}
	if h := got.header.Get("X-Title"); h != "goeval-test" {
		t.Errorf("X-Title = %q", h)
	}
	if got.body["model"] != "openai/gpt-test" {
		t.Errorf("model = %v", got.body["model"])
	}
	msgs := got.body["messages"].([]any)
	first := msgs[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "hello" {
		t.Errorf("messages = %v", msgs)
	}
	rf := got.body["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v", rf["type"])
	}
	js := rf["json_schema"].(map[string]any)
	if js["name"] == "" || js["name"] == nil {
		t.Errorf("json_schema.name missing")
	}
	if js["strict"] != true {
		t.Errorf("json_schema.strict = %v", js["strict"])
	}
	schema := js["schema"].(map[string]any)
	if schema["type"] != "object" || schema["additionalProperties"] != false {
		t.Errorf("json_schema.schema = %v", schema)
	}
	if p, ok := got.body["provider"].(map[string]any); !ok || p["require_parameters"] != true {
		t.Errorf("extra body not merged: %v", got.body["provider"])
	}
}

func TestNoKeySendsNoAuthorization(t *testing.T) {
	srv, got := newServer(t, 200, chatBody(`{"choice":"A"}`))
	g := NewGenerator("m", WithBaseURL(srv.URL))
	if _, err := g.StructuredGenerate(context.Background(), "p", testSchema); err != nil {
		t.Fatal(err)
	}
	if _, ok := got.header["Authorization"]; ok {
		t.Errorf("Authorization sent without a key")
	}
}

func TestDefaultBaseURL(t *testing.T) {
	g := NewGenerator("m")
	if g.opts.baseURL != DefaultBaseURL {
		t.Errorf("baseURL = %q", g.opts.baseURL)
	}
	if g.opts.timeout != DefaultTimeout {
		t.Errorf("timeout = %v", g.opts.timeout)
	}
	g = NewGenerator("m", WithBaseURL(""), WithTimeout(-1))
	if g.opts.baseURL != DefaultBaseURL || g.opts.timeout != DefaultTimeout {
		t.Errorf("empty options not defaulted: %q %v", g.opts.baseURL, g.opts.timeout)
	}
}

func TestStructuredGenerateErrors(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		wantStatus int
		wantSubstr string
	}{
		{name: "unauthorized", status: 401, body: `{"error":{"message":"bad key"}}`, wantStatus: 401, wantSubstr: "bad key"},
		{name: "server error", status: 500, body: "boom", wantStatus: 500, wantSubstr: "boom"},
		{name: "error body echoes key", status: 401, body: "Incorrect API key provided: " + testKey, wantStatus: 401, wantSubstr: "[REDACTED]"},
		{name: "long body trimmed", status: 502, body: strings.Repeat("x", 5000), wantStatus: 502, wantSubstr: "(truncated)"},
		{name: "error object with 200", status: 200, body: `{"error":{"message":"upstream down"}}`, wantSubstr: "upstream down"},
		{name: "no choices", status: 200, body: `{"choices":[]}`, wantSubstr: "no choices"},
		{name: "refusal", status: 200, body: `{"choices":[{"message":{"content":"","refusal":"cannot help"}}]}`, wantSubstr: "cannot help"},
		{name: "empty content", status: 200, body: chatBody(""), wantSubstr: "empty content"},
		{name: "not json content", status: 200, body: chatBody("sure! here you go"), wantSubstr: "failed to parse JSON"},
		{name: "not an object", status: 200, body: chatBody(`["A"]`), wantSubstr: "failed to parse JSON"},
		{name: "reply not json", status: 200, body: "<html>", wantSubstr: "failed to parse response"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := newServer(t, tt.status, tt.body)
			g := NewGenerator("m", WithBaseURL(srv.URL), WithAPIKey(testKey))
			_, err := g.StructuredGenerate(context.Background(), "p", testSchema)
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, api.ErrLLMGenerationFailed) {
				t.Errorf("error does not wrap ErrLLMGenerationFailed: %v", err)
			}
			if !strings.Contains(err.Error(), tt.wantSubstr) {
				t.Errorf("error %q lacks %q", err, tt.wantSubstr)
			}
			if strings.Contains(err.Error(), testKey) {
				t.Errorf("error leaks the API key: %v", err)
			}
			var apiErr *APIError
			if tt.wantStatus != 0 {
				if !errors.As(err, &apiErr) || apiErr.StatusCode != tt.wantStatus {
					t.Errorf("want APIError status %d, got %v", tt.wantStatus, err)
				}
			}
		})
	}
}

func TestTransportErrorDoesNotLeakKey(t *testing.T) {
	// A URL that carries the key in its userinfo would surface in net/http errors
	g := NewGenerator("m", WithBaseURL("http://127.0.0.1:1/"+testKey), WithAPIKey(testKey), WithTimeout(time.Second))
	_, err := g.StructuredGenerate(context.Background(), "p", testSchema)
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), testKey) {
		t.Errorf("error leaks the API key: %v", err)
	}
	if !errors.Is(err, api.ErrLLMGenerationFailed) {
		t.Errorf("error does not wrap ErrLLMGenerationFailed: %v", err)
	}
}

func TestTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(300 * time.Millisecond):
		}
	}))
	t.Cleanup(srv.Close)
	g := NewGenerator("m", WithBaseURL(srv.URL), WithTimeout(50*time.Millisecond))
	if _, err := g.StructuredGenerate(context.Background(), "p", testSchema); err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestCustomHTTPClientIsUsed(t *testing.T) {
	srv, _ := newServer(t, 200, chatBody(`{"choice":"A"}`))
	used := false
	hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		used = true
		return http.DefaultTransport.RoundTrip(r)
	})}
	g := NewGenerator("m", WithBaseURL(srv.URL), WithHTTPClient(hc))
	if _, err := g.StructuredGenerate(context.Background(), "p", testSchema); err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Error("custom client not used")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestEmbed(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		body    string
		want    []float64
		wantErr string
	}{
		{name: "normalized", status: 200, body: `{"data":[{"embedding":[3,4]}]}`, want: []float64{0.6, 0.8}},
		{name: "no data", status: 200, body: `{"data":[]}`, wantErr: "no embeddings"},
		{name: "empty vector", status: 200, body: `{"data":[{"embedding":[]}]}`, wantErr: "empty embedding"},
		{name: "zero vector", status: 200, body: `{"data":[{"embedding":[0,0]}]}`, wantErr: "zero embedding"},
		{name: "http error", status: 429, body: `rate limited`, wantErr: "HTTP 429"},
		{name: "error object", status: 200, body: `{"error":{"message":"no such model"}}`, wantErr: "no such model"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, got := newServer(t, tt.status, tt.body)
			e := NewEmbedder("text-embedding-test", WithBaseURL(srv.URL), WithAPIKey(testKey))
			vec, err := e.Embed(context.Background(), "hi")
			if got.path != "/embeddings" {
				t.Errorf("path = %q", got.path)
			}
			if got.body["model"] != "text-embedding-test" || got.body["input"] != "hi" {
				t.Errorf("body = %v", got.body)
			}
			if got.header.Get("Authorization") != "Bearer "+testKey {
				t.Errorf("Authorization = %q", got.header.Get("Authorization"))
			}
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(vec) != len(tt.want) {
				t.Fatalf("len = %d", len(vec))
			}
			for i := range vec {
				if math.Abs(vec[i]-tt.want[i]) > 1e-9 {
					t.Errorf("vec[%d] = %v, want %v", i, vec[i], tt.want[i])
				}
			}
		})
	}
}

func TestGeneratorTimeoutIsDetectable(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release) // runs first: frees the handler so Close doesn't wait

	gen := NewGenerator("m", WithBaseURL(srv.URL), WithAPIKey("test-key-not-real-123"), WithTimeout(50*time.Millisecond))
	_, err := gen.StructuredGenerate(context.Background(), "p", map[string]interface{}{"type": "object"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want context.DeadlineExceeded in chain, got %v", err)
	}
	if strings.Contains(err.Error(), "test-key-not-real-123") {
		t.Fatalf("error leaks the key: %v", err)
	}
}
