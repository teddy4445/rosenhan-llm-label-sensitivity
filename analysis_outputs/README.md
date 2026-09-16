# Analysis outputs

Summary results for Study 1 (5 models × 2 psychiatric-background conditions × 2 current-evidence conditions × 1,000 calls = 20,000 responses). Each file below was computed from the response-level data.

Condition labels used in every file:

- `current_evidence`: `Voices ceased` (the reference disposition is discharge) or `Voices persisted` (the reference disposition is continued hospitalization)
- `prior_schizophrenia`: `No` (no prior psychiatric diagnosis stated) or `Yes` (prior schizophrenia diagnosis documented)

## Disposition

| File | Contents |
|---|---|
| `model_cell_results.csv` | Hospitalization recommendations (n and %) for each of the 20 model × condition cells |
| `pooled_disposition_results.csv` | The same counts pooled across the five models |
| `pooled_disposition_effects.csv` | Pooled label effect in each evidence condition: risk difference (Newcombe 95% CI) and risk ratio |
| `model_specific_label_effects.csv` | Label effect per model and evidence condition: risk difference with Newcombe 95% CI, risk ratio, odds ratio with 95% CI, two-sided Fisher's exact p, and Holm-adjusted p across the 10 contrasts |
| `reference_concordance_results.csv` | Pooled agreement with the prespecified reference disposition |

## Confidence (0–100)

| File | Contents |
|---|---|
| `confidence_cell_results.csv` | Mean, SD and median confidence for each of the 20 cells |
| `confidence_label_effects.csv` | Label effect per model and evidence condition: mean difference with 95% CI, Cohen's d, Welch t and df, p, and Holm-adjusted p |
| `confidence_contrasts.csv` | The same contrasts with formatted column names |
| `supplementary_table_confidence_contrasts.csv` | Identical to `confidence_contrasts.csv` |
| `table_3_confidence_effects.csv` | Formatted mean differences and Cohen's d for the manuscript table |
| `confidence_regression_results.csv` | Label effect from linear regression with HC3 standard errors, adjusted for evidence condition and model, and then also adjusted for the disposition recommendation |

## Signal detection

Hit = recommending hospitalization when the voices persisted. False alarm = recommending hospitalization when the voices had ceased.

| File | Contents |
|---|---|
| `signal_detection_indices.csv` | Hit rate, false-alarm rate, d′ and criterion c for each model in each background condition |
| `signal_detection_changes.csv` | Change in d′ and c when the prior diagnosis is added (label minus no label) |
