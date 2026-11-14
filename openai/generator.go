package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/datar-psa/goeval/api"
)

// Generator implements api.LLMGenerator on the Chat Completions API
type Generator struct {
	client
}

// NewGenerator creates a generator for an OpenAI-compatible server
// model: the model id (e.g., "gpt-4o-mini", or "openai/gpt-4o-mini" on OpenRouter)
func NewGenerator(model string, opts ...Option) *Generator {
	return &Generator{client: newClient(model, opts)}
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			Refusal string `json:"refusal"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// StructuredGenerate implements LLMGenerator.StructuredGenerate using a strict json_schema response format.
// Note: strict mode requires the schema to set additionalProperties:false and list every property as required.
func (g *Generator) StructuredGenerate(ctx context.Context, prompt string, schema map[string]interface{}) (map[string]interface{}, error) {
	body := map[string]any{
		"model": g.model,
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"schema": schema,
				"strict": true,
			},
		},
	}

	var resp chatResponse
	if err := g.post(ctx, "/chat/completions", body, &resp); err != nil {
		return nil, fmt.Errorf("%w: %w", api.ErrLLMGenerationFailed, err)
	}

	// Some servers answer 200 with an error object (OpenRouter does for upstream failures)
	if resp.Error != nil {
		return nil, fmt.Errorf("%w: provider error: %s", api.ErrLLMGenerationFailed, g.trim([]byte(resp.Error.Message)))
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("%w: no choices returned", api.ErrLLMGenerationFailed)
	}
	msg := resp.Choices[0].Message
	if msg.Refusal != "" {
		return nil, fmt.Errorf("%w: model refused: %s", api.ErrLLMGenerationFailed, g.trim([]byte(msg.Refusal)))
	}
	if msg.Content == "" {
		return nil, fmt.Errorf("%w: empty content (finish_reason %q)", api.ErrLLMGenerationFailed, resp.Choices[0].FinishReason)
	}

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(msg.Content), &result); err != nil {
		return nil, fmt.Errorf("%w: failed to parse JSON response: %w, response: %s", api.ErrLLMGenerationFailed, err, g.trim([]byte(msg.Content)))
	}
	return result, nil
}

// Verify that Generator implements LLMGenerator
var _ api.LLMGenerator = (*Generator)(nil)
