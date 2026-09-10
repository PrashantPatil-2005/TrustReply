# Golden Set Labeling Protocol

## Sampling (n=200)

1. **Source**: Real AmazonHelp customer→brand reply pairs from the Kaggle TWCS dataset (Oct 2017).
2. **Extraction**: Pairs where AmazonHelp replied to a customer tweet mentioning @AmazonHelp.
3. **Stratification**: ~20 examples per intent class (10 intents → 200 total).
4. **Seed**: `random_seed=42` for reproducibility.

## Intent Labels (10 classes)

| Intent | Definition | Example trigger phrases |
|--------|-----------|------------------------|
| `order_status` | Tracking, shipping timing | "where is my order", "tracking" |
| `delivery_issue` | Late/lost/damaged delivery | "never arrived", "wrong address" |
| `refund_return` | Returns, refunds, exchanges | "money back", "return this" |
| `account_access` | Login/password issues | "can't log in", "reset password" |
| `payment_billing` | Charges, billing disputes | "charged twice", "unauthorized" |
| `product_quality` | Defective/broken products | "not working", "broken" |
| `subscription_prime` | Prime membership | "cancel prime", "prime charge" |
| `technical_app` | App/device/website issues | "kindle", "fire tv", "app crash" |
| `praise_thanks` | Positive feedback | "thank you", "great service" |
| `general_inquiry` | Unclassified help requests | "help me", "question" |

**Labeling process**: Rule-based pre-label → human review disambiguation rules (see `build_golden_set.py::_human_review_intent`).

## Escalation Labels

| Label | Criteria |
|-------|----------|
| `auto_handle` | Praise/thanks; simple order status; low-risk general inquiries |
| `escalate` | Payment disputes, fraud/legal keywords, high frustration, refunds, account security, product issues, delivery failures |

**Conservative bias**: Financial and product intents default to `escalate` unless clearly low-risk.

## Human Reply Quality (1–5)

Scored for LLM-judge agreement study:
- 5: Empathetic, actionable, on-brand
- 3: Adequate but generic
- 1: Unhelpful or off-topic

## Known Limitations

- Labels reflect Oct 2017 AmazonHelp practices; may not match 2026 policies.
- English-only; Japanese/other-language tweets excluded from golden set.
- Single annotator (author) with documented guidelines; no inter-annotator agreement computed.
