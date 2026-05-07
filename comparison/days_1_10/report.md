# Comparison report: baseline vs grounded

## Slack

| metric | baseline | grounded | delta |
|---|---|---|---|
| count | 379 | 1029 | ↑ 171.5% (a=379, b=1029) |
| char_mean | 125 | 273 | ↑ 118.4% (a=125, b=273) |
| char_p95 | 249 | 399 | ↑ 60.2% (a=249, b=399) |
| word_mean | 20 | 43 | ↑ 115.0% (a=20, b=43) |
| word_p95 | 39 | 60 | ↑ 53.8% (a=39, b=60) |
| vagueness_pct | 10.82 | 21.67 | ↑ 100.3% (a=10.82, b=21.67) |
| drift_pct | 0.0 | 0.39 | new (0.39) |
| mention_rate_pct | 0.0 | 0.0 | no change |
| quote_rate_pct | 0.0 | 0.0 | no change |
| code_block_rate_pct | 0.0 | 0.0 | no change |
| emoji_rate_pct | 0.0 | 0.49 | new (0.49) |

## Email

| metric | baseline | grounded | delta |
|---|---|---|---|
| count | 295 | 172 | ↓ 41.7% (a=295, b=172) |
| char_mean | 363 | 429 | ↑ 18.2% (a=363, b=429) |
| char_p95 | 653 | 636 | ↓ 2.6% (a=653, b=636) |
| word_mean | 57 | 67 | ↑ 17.5% (a=57, b=67) |
| word_p95 | 99 | 97 | ↓ 2.0% (a=99, b=97) |
| vagueness_pct | 0.34 | 4.65 | ↑ 1267.6% (a=0.34, b=4.65) |
| drift_pct | 0.0 | 0.0 | no change |
| mention_rate_pct | 0.0 | 0.0 | no change |
| quote_rate_pct | 0.0 | 0.0 | no change |
| code_block_rate_pct | 0.0 | 0.0 | no change |
| emoji_rate_pct | 0.0 | 0.0 | no change |

## Interpretation hints

- `vagueness_pct` rising in the grounded run is a positive signal — real workplace chat carries hedges ("i think", "let me check") that pure-synthetic LLM output tends to smooth away.
- `drift_pct` rising is a positive signal for chat realism (off-topic chatter is real). For email it should stay low.
- `char_p95` shrinking and `mention_rate_pct` rising in the grounded Slack run = closer to real channel patterns (shorter messages, more @-tags).
