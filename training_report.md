# AltLAS Training Report

<!-- 
This is an auto-generated report summarizing the latest training cycle statistics.
Last updated: {TIMESTAMP}
-->

## Run Metadata

| Metric | Value |
|--------|-------|
| Report Generation Time | {TIMESTAMP} |
| Training Session ID | {SESSION_ID} |
| Task Name | {TASK_NAME} |
| Total Attempts | {TOTAL_ATTEMPTS} |
| Successful Attempts | {SUCCESSFUL_ATTEMPTS} |
| Success Rate | {SUCCESS_RATE}% |
| Total Training Time | {TRAINING_TIME} |

## Performance Metrics

### Score Progression

- **Current Moving Average Score**: {MOVING_AVG_SCORE}
- **Score Trend**: {SCORE_TREND} <!-- e.g., "Increasing", "Plateaued", "Declining" -->
- **Learning Rate**: {LEARNING_RATE}
- **Entropy Coefficient**: {ENTROPY_COEFFICIENT}

### Best and Worst Attempts

#### Highest Scoring Attempt (Score: {HIGH_SCORE})

```python
{HIGHEST_SCORING_CODE}
```

**Output**:
```
{HIGHEST_SCORING_OUTPUT}
```

#### Lowest Scoring Attempt (Score: {LOW_SCORE})

```python
{LOWEST_SCORING_CODE}
```

**Output**:
```
{LOWEST_SCORING_OUTPUT}
```

## Pattern Analysis

### Common Patterns in Successful Attempts

<!-- List of code patterns that appear frequently in high-scoring solutions -->
1. {PATTERN_1}
2. {PATTERN_2}
3. {PATTERN_3}

### Common Patterns in Failed Attempts

<!-- List of code patterns that appear frequently in low-scoring solutions -->
1. {PATTERN_1}
2. {PATTERN_2}
3. {PATTERN_3}

### Common Errors

| Error Type | Frequency | Example |
|------------|-----------|---------|
| {ERROR_TYPE_1} | {COUNT_1} | `{ERROR_EXAMPLE_1}` |
| {ERROR_TYPE_2} | {COUNT_2} | `{ERROR_EXAMPLE_2}` |
| {ERROR_TYPE_3} | {COUNT_3} | `{ERROR_EXAMPLE_3}` |

## Learning Status

### Performance Indicators

- **Plateau Detection**: {PLATEAU_STATUS} <!-- e.g., "None detected", "Potential plateau at score 0.45" -->
- **Semantic Drift**: {SEMANTIC_DRIFT_STATUS} <!-- e.g., "None detected", "Moderate drift from original task objective" -->
- **Token Distribution**: {TOKEN_DISTRIBUTION_STATUS} <!-- e.g., "Healthy", "Skewed towards basic syntax tokens" -->
- **Hint Usage**: {HINT_USAGE} hints provided (utilized in {HINT_UTILIZATION}% of subsequent attempts)
- **Average Hint Improvement**: {AVG_HINT_IMPROVEMENT} <!-- Average score change in the attempt immediately following a hint -->

### Score Component Breakdown

| Component | Weight | Average Value |
|-----------|--------|--------------|
| Syntax | {SYNTAX_WEIGHT} | {SYNTAX_SCORE} |
| Execution | {EXECUTION_WEIGHT} | {EXECUTION_SCORE} |
| Output | {OUTPUT_WEIGHT} | {OUTPUT_SCORE} |
| Structural | {STRUCTURAL_WEIGHT} | {STRUCTURAL_SCORE} |
| Constraints | {CONSTRAINTS_WEIGHT} | {CONSTRAINTS_SCORE} |
| Semantic | {SEMANTIC_WEIGHT} | {SEMANTIC_SCORE} |

## LLM Provider Performance

| Metric                     | Value                       |
|----------------------------|-----------------------------|
| Total Requests             | {LLM_TOTAL_REQUESTS}        |
| Failed Requests            | {LLM_FAILED_REQUESTS}       |
| Avg Latency (Overall)    | {LLM_AVG_LATENCY}           |
| Avg Latency (Recent)     | {LLM_AVG_RECENT_LATENCY}    |
| Total Prompt Tokens        | {LLM_TOTAL_PROMPT_TOKENS}   |
| Total Response Tokens      | {LLM_TOTAL_RESPONSE_TOKENS} |
| Total Tokens               | {LLM_TOTAL_TOKENS}          |
| Avg Prompt Tokens/Req    | {LLM_AVG_PROMPT_TOKENS}     |
| Avg Response Tokens/Req  | {LLM_AVG_RESPONSE_TOKENS}   |
| Cache Hit Rate           | {LLM_CACHE_HIT_RATE}        |
| Cache Miss Rate          | {LLM_CACHE_MISS_RATE}       |
| Cache Size (Current/Cap) | {LLM_CACHE_SIZE}            |
| Single Requests Processed  | {LLM_SINGLE_REQUESTS}       |
| Batched Requests Processed | {LLM_BATCHED_REQUESTS}      |
| Batches Processed          | {LLM_BATCHES_PROCESSED}     |
| Async Queue Max Observed   | {LLM_ASYNC_QUEUE_MAX}       |
| Batching Enabled           | {LLM_BATCHING_ENABLED}      |
| Async Enabled              | {LLM_ASYNC_ENABLED}         |


## Resource Utilization

- **Memory Usage**: {MEMORY_USAGE}
- **Average Attempt Generation Time**: {AVG_GEN_TIME}ms
- **Average Execution Time**: {AVG_EXEC_TIME}ms

## Notes and Observations

<!-- 
This section can be auto-populated with AI-generated observations or manually updated.
It should contain insights about the training process, suggestions for improvement,
or noteworthy patterns that might inform future task adjustments.
-->

{OBSERVATIONS}

### Recommendations

1. {RECOMMENDATION_1}
2. {RECOMMENDATION_2}
3. {RECOMMENDATION_3}

---

## Historical Comparison

| Metric | Previous Report | Current Report | Change |
|--------|----------------|----------------|--------|
| Success Rate | {PREV_SUCCESS_RATE}% | {SUCCESS_RATE}% | {SUCCESS_RATE_CHANGE}% |
| Moving Avg Score | {PREV_MOVING_AVG} | {MOVING_AVG_SCORE} | {SCORE_CHANGE} |
| Best Score | {PREV_HIGH_SCORE} | {HIGH_SCORE} | {HIGH_SCORE_CHANGE} |
| Entropy Coefficient | {PREV_ENTROPY} | {ENTROPY_COEFFICIENT} | {ENTROPY_CHANGE} |

---

<!-- 
Additional fields for developer use - not displayed in the report UI
{CUSTOM_FIELD_1}
{CUSTOM_FIELD_2}
{CUSTOM_FIELD_3}
-->