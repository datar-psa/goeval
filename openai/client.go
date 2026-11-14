// Package openai provides an OpenAI-compatible Chat Completions provider.
//
// It works against any server that speaks the OpenAI wire protocol: OpenAI
// itself (the default), OpenRouter (https://openrouter.ai/api/v1), Ollama,
// vLLM and LM Studio. It uses only net/http from the standard library.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the OpenAI API root. OpenRouter is "https://openrouter.ai/api/v1".
	DefaultBaseURL = "https://api.openai.com/v1"
	// DefaultTimeout is the per-request timeout when none is configured.
	DefaultTimeout = 60 * time.Second

	maxResponseBytes = 8 << 20
	maxErrorBodyLen  = 512
)

// Options configures a Generator or an Embedder
type Options struct {
	baseURL    string
	apiKey     string
	headers    map[string]string
	httpClient *http.Client
	timeout    time.Duration
	extraBody  map[string]any
}

// Option customizes Options
type Option func(*Options)

// WithBaseURL sets the API root, without a trailing path such as /chat/completions
func WithBaseURL(baseURL string) Option {
	return func(o *Options) { o.baseURL = baseURL }
}

// WithAPIKey sets the bearer token. Leave it unset for servers that need no key (e.g. Ollama).
func WithAPIKey(apiKey string) Option {
	return func(o *Options) { o.apiKey = apiKey }
}

// WithHeader adds a request header, e.g. OpenRouter's HTTP-Referer or X-Title.
// Authorization and Content-Type are set by the provider and cannot be overridden.
func WithHeader(name, value string) Option {
	return func(o *Options) {
		if o.headers == nil {
			o.headers = make(map[string]string)
		}
		o.headers[name] = value
	}
}

// WithHTTPClient sets the HTTP client, e.g. one wrapped for recording or tracing
func WithHTTPClient(client *http.Client) Option {
	return func(o *Options) { o.httpClient = client }
}

// WithTimeout sets the per-request timeout (default DefaultTimeout)
func WithTimeout(timeout time.Duration) Option {
	return func(o *Options) { o.timeout = timeout }
}

// WithExtraBody merges top-level fields into every request body, for provider
// extensions such as OpenRouter's {"provider": {"require_parameters": true}}.
// Fields the provider sets itself (model, messages, response_format, input) win.
func WithExtraBody(fields map[string]any) Option {
	return func(o *Options) {
		if o.extraBody == nil {
			o.extraBody = make(map[string]any)
		}
		for k, v := range fields {
			o.extraBody[k] = v
		}
	}
}

// APIError is returned for a non-2xx response
type APIError struct {
	// StatusCode is the HTTP status code
	StatusCode int
	// Body is the provider's error body, trimmed and with the API key removed
	Body string
}

func (e *APIError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("openai: HTTP %d", e.StatusCode)
	}
	return fmt.Sprintf("openai: HTTP %d: %s", e.StatusCode, e.Body)
}

type client struct {
	opts  Options
	model string
}

func newClient(model string, opts []Option) client {
	o := Options{baseURL: DefaultBaseURL, timeout: DefaultTimeout}
	for _, opt := range opts {
		opt(&o)
	}
	if o.baseURL == "" {
		o.baseURL = DefaultBaseURL
	}
	o.baseURL = strings.TrimRight(o.baseURL, "/")
	if o.timeout <= 0 {
		o.timeout = DefaultTimeout
	}
	if o.httpClient == nil {
		o.httpClient = &http.Client{}
	}
	return client{opts: o, model: model}
}

// redact removes the API key from text that may reach an error
func (c client) redact(s string) string {
	if c.opts.apiKey == "" {
		return s
	}
	return strings.ReplaceAll(s, c.opts.apiKey, "[REDACTED]")
}

// post sends body as JSON to path and decodes a 2xx reply into out
func (c client) post(ctx context.Context, path string, body map[string]any, out any) error {
	for k, v := range c.opts.extraBody {
		if _, set := body[k]; !set {
			body[k] = v
		}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, c.opts.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.opts.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return errors.New(c.redact(fmt.Sprintf("failed to create request: %v", err)))
	}
	for k, v := range c.opts.headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.opts.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.opts.apiKey)
	}

	resp, err := c.opts.httpClient.Do(req)
	if err != nil {
		// A timeout or cancellation stays detectable with errors.Is; its text holds no key
		if ctxErr := ctx.Err(); ctxErr != nil {
			return fmt.Errorf("request failed: %w", ctxErr)
		}
		// Rebuilt without wrapping so nothing but redacted text escapes
		return errors.New(c.redact(fmt.Sprintf("request failed: %v", err)))
	}
	defer func() { _ = resp.Body.Close() }()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return errors.New(c.redact(fmt.Sprintf("failed to read response: %v", err)))
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return &APIError{StatusCode: resp.StatusCode, Body: c.trim(raw)}
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("failed to parse response: %w, response: %s", err, c.trim(raw))
	}
	return nil
}

func (c client) trim(raw []byte) string {
	s := strings.TrimSpace(c.redact(string(raw)))
	if len(s) > maxErrorBodyLen {
		s = s[:maxErrorBodyLen] + "...(truncated)"
	}
	return s
}
