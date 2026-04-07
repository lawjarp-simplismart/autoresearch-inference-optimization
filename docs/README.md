# Reference Docs

Model-specific and framework-specific reference material for the agent.

## Structure

```
docs/
  <model-name>/              # one folder per model
    vllm_recipe.md           # vLLM official recipe/guide
    sglang_recipe.md         # SGLang guide (if applicable)
    notes.md                 # agent's own findings/notes
```

## Adding docs for a new model

1. Create a folder: `docs/<model-short-name>/`
2. Add framework recipes (from official docs)
3. The agent reads these before experimenting with unfamiliar flags
