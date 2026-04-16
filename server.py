#!/usr/bin/env python3
"""
AI Explainability Report MCP Server
=====================================
By MEOK AI Labs | https://meok.ai

The only MCP server for AI explainability and transparency reports.
Covers model cards, decision explanations, transparency audits,
and impact assessments per EU AI Act Article 13 and NIST AI RMF.

Install: pip install mcp
Run:     python server.py
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from mcp.server.fastmcp import FastMCP

# -- Authentication --------------------------------------------------------
import os as _os
import sys, os

_MEOK_API_KEY = _os.environ.get("MEOK_API_KEY", "")

try:
    sys.path.insert(0, os.path.expanduser("~/clawd/meok-labs-engine/shared"))
    from auth_middleware import check_access as _shared_check_access
    _AUTH_ENGINE_AVAILABLE = True
except ImportError:
    _AUTH_ENGINE_AVAILABLE = False

    def _shared_check_access(api_key=""):
        # type: (str) -> Tuple[bool, str, str]
        """Fallback when shared auth engine is not available."""
        if _MEOK_API_KEY and api_key and api_key == _MEOK_API_KEY:
            return True, "OK", "pro"
        if _MEOK_API_KEY and api_key and api_key != _MEOK_API_KEY:
            return False, "Invalid API key. Get one at https://meok.ai/api-keys", "free"
        return True, "OK", "free"


def check_access(api_key=""):
    # type: (str) -> Tuple[bool, str, str]
    """Unified access check -- works with or without shared auth engine."""
    return _shared_check_access(api_key)


# -- Rate limiting ---------------------------------------------------------
FREE_DAILY_LIMIT = 10
PRO_TIER_UNLIMITED = True  # Pro: $29/mo unlimited at https://meok.ai/mcp/explainability-report/pro
_usage = defaultdict(list)  # type: Dict[str, List[datetime]]


def _check_rate_limit(caller="anonymous", tier="free"):
    # type: (str, str) -> Optional[str]
    """Returns error string if rate-limited, else None."""
    if tier == "pro":
        return None
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    _usage[caller] = [t for t in _usage[caller] if t > cutoff]
    if len(_usage[caller]) >= FREE_DAILY_LIMIT:
        return (
            "Free tier limit reached ({}/day). "
            "Upgrade to MEOK AI Labs Pro for unlimited access at $29/mo: "
            "https://meok.ai/mcp/explainability-report/pro".format(FREE_DAILY_LIMIT)
        )
    _usage[caller].append(now)
    return None


# ---------------------------------------------------------------------------
# Transparency Knowledge Base
# ---------------------------------------------------------------------------

TRANSPARENCY_REQUIREMENTS = {
    "article_13": {
        "title": "Transparency and Provision of Information to Deployers",
        "checks": [
            {"id": "T1", "requirement": "Instructions for use accompany the AI system", "category": "documentation"},
            {"id": "T2", "requirement": "Identity and contact details of the provider are included", "category": "identification"},
            {"id": "T3", "requirement": "Characteristics, capabilities, and limitations are described", "category": "capabilities"},
            {"id": "T4", "requirement": "Intended purpose and any preconditions are specified", "category": "purpose"},
            {"id": "T5", "requirement": "Level of accuracy and accuracy metrics are declared", "category": "performance"},
            {"id": "T6", "requirement": "Known or foreseeable circumstances affecting performance are documented", "category": "limitations"},
            {"id": "T7", "requirement": "Human oversight measures are described per Article 14", "category": "oversight"},
            {"id": "T8", "requirement": "Expected lifetime and maintenance requirements specified", "category": "lifecycle"},
            {"id": "T9", "requirement": "Computational and hardware resource requirements documented", "category": "technical"},
            {"id": "T10", "requirement": "Relevant input data specifications are provided", "category": "data"},
            {"id": "T11", "requirement": "Information enabling deployers to interpret output is provided", "category": "interpretability"},
            {"id": "T12", "requirement": "Changes to the system during its lifecycle are documented", "category": "versioning"},
        ],
    },
    "article_50": {
        "title": "Transparency Obligations for Certain AI Systems",
        "checks": [
            {"id": "A50-1", "requirement": "Users are informed they are interacting with an AI system", "category": "disclosure"},
            {"id": "A50-2", "requirement": "AI-generated content is marked as artificially generated or manipulated", "category": "labelling"},
            {"id": "A50-3", "requirement": "Emotion recognition or biometric categorisation systems inform the person", "category": "biometric_notice"},
            {"id": "A50-4", "requirement": "Deep fakes are labelled indicating artificial generation", "category": "deepfake_label"},
        ],
    },
}

EXPLAINABILITY_METHODS = {
    "feature_importance": {
        "name": "Feature Importance / Attribution",
        "description": "Ranks input features by their contribution to the prediction",
        "techniques": ["SHAP", "LIME", "Integrated Gradients", "Permutation Importance"],
        "best_for": ["tabular data", "tree models", "neural networks"],
        "limitations": ["May not capture feature interactions", "Can be unstable for correlated features"],
    },
    "counterfactual": {
        "name": "Counterfactual Explanations",
        "description": "Shows what minimal changes to inputs would change the prediction",
        "techniques": ["DiCE", "Alibi Counterfactuals", "CARLA"],
        "best_for": ["individual decisions", "recourse", "credit/loan decisions"],
        "limitations": ["May suggest impractical changes", "Multiple valid counterfactuals exist"],
    },
    "rule_extraction": {
        "name": "Rule Extraction",
        "description": "Extracts human-readable rules that approximate model behaviour",
        "techniques": ["Decision Tree Surrogate", "Anchors", "Rule Lists"],
        "best_for": ["compliance reporting", "policy documentation", "auditing"],
        "limitations": ["Approximation may miss edge cases", "Trade-off between fidelity and simplicity"],
    },
    "attention_visualization": {
        "name": "Attention / Saliency Maps",
        "description": "Visualises which parts of the input the model focused on",
        "techniques": ["Attention Weights", "GradCAM", "Saliency Maps", "Layer-wise Relevance Propagation"],
        "best_for": ["NLP models", "image classification", "medical imaging"],
        "limitations": ["Attention may not equal importance", "Can be manipulated adversarially"],
    },
    "concept_based": {
        "name": "Concept-Based Explanations",
        "description": "Explains predictions in terms of human-understandable concepts",
        "techniques": ["TCAV", "Concept Bottleneck Models", "Network Dissection"],
        "best_for": ["image models", "domain expert communication", "high-stakes decisions"],
        "limitations": ["Requires concept labelling", "Limited to pre-defined concepts"],
    },
    "example_based": {
        "name": "Example-Based Explanations",
        "description": "Explains predictions by showing similar training examples",
        "techniques": ["Prototypes", "Influential Instances", "k-NN Explanations"],
        "best_for": ["case-based reasoning", "medical diagnosis", "legal precedent"],
        "limitations": ["Requires representative training data access", "Privacy concerns with showing training data"],
    },
}

MODEL_TYPES = {
    "classification": {
        "metrics": ["accuracy", "precision", "recall", "F1-score", "AUC-ROC", "confusion matrix"],
        "explainability_methods": ["feature_importance", "counterfactual", "rule_extraction"],
    },
    "regression": {
        "metrics": ["MAE", "MSE", "RMSE", "R-squared", "MAPE"],
        "explainability_methods": ["feature_importance", "counterfactual"],
    },
    "nlp": {
        "metrics": ["BLEU", "ROUGE", "perplexity", "accuracy", "F1-score"],
        "explainability_methods": ["attention_visualization", "feature_importance", "example_based"],
    },
    "computer_vision": {
        "metrics": ["accuracy", "mAP", "IoU", "precision", "recall"],
        "explainability_methods": ["attention_visualization", "concept_based", "example_based"],
    },
    "generative": {
        "metrics": ["FID", "IS", "BLEU", "human evaluation", "toxicity score"],
        "explainability_methods": ["attention_visualization", "concept_based"],
    },
    "recommendation": {
        "metrics": ["precision@k", "recall@k", "NDCG", "diversity", "coverage"],
        "explainability_methods": ["feature_importance", "example_based", "rule_extraction"],
    },
}

RISK_CATEGORIES = {
    "fundamental_rights": {
        "name": "Fundamental Rights Impact",
        "areas": [
            "Right to non-discrimination (Charter Art. 21)",
            "Right to privacy and data protection (Charter Art. 7, 8)",
            "Freedom of expression and information (Charter Art. 11)",
            "Right to an effective remedy (Charter Art. 47)",
            "Presumption of innocence (Charter Art. 48)",
            "Rights of the child (Charter Art. 24)",
            "Rights of persons with disabilities (Charter Art. 26)",
        ],
    },
    "safety": {
        "name": "Safety Impact",
        "areas": [
            "Physical safety of persons",
            "Psychological well-being",
            "Environmental impact",
            "Property damage potential",
            "Public infrastructure safety",
        ],
    },
    "societal": {
        "name": "Societal Impact",
        "areas": [
            "Democratic processes and civic participation",
            "Social cohesion and equality",
            "Access to essential services",
            "Labour market effects",
            "Cultural and linguistic diversity",
            "Misinformation and manipulation risk",
        ],
    },
}


def _match_keywords(text, keywords):
    # type: (str, List[str]) -> List[str]
    """Return matched keywords found in text (case-insensitive)."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def _assess_transparency_level(description):
    # type: (str) -> Tuple[str, float, List[str]]
    """Assess how transparent an AI system appears from its description."""
    text_lower = description.lower()
    score = 0.0
    positive_signals = []  # type: List[str]
    max_score = 12.0

    transparency_indicators = [
        ("document", "Documentation mentioned"),
        ("explain", "Explainability considered"),
        ("transparen", "Transparency explicitly addressed"),
        ("human oversight", "Human oversight mentioned"),
        ("audit", "Auditability considered"),
        ("log", "Logging capability mentioned"),
        ("monitor", "Monitoring mentioned"),
        ("bias", "Bias awareness mentioned"),
        ("fairness", "Fairness considered"),
        ("accuracy", "Accuracy metrics mentioned"),
        ("limitation", "Limitations acknowledged"),
        ("user inform", "User information provided"),
    ]

    for keyword, signal in transparency_indicators:
        if keyword in text_lower:
            score += 1.0
            positive_signals.append(signal)

    normalised = score / max_score
    if normalised >= 0.6:
        level = "high"
    elif normalised >= 0.3:
        level = "moderate"
    else:
        level = "low"

    return level, round(normalised, 2), positive_signals


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "AI Explainability Report",
    instructions=(
        "By MEOK AI Labs -- AI explainability and transparency reports. "
        "Start with quick_scan (one sentence, instant transparency assessment). "
        "Full tools: generate_model_card, explain_decision, transparency_audit, create_impact_assessment. "
        "No API key needed for free tier (10 calls/day)."
    ),
)


# ---------------------------------------------------------------------------
# Tool: quick_scan -- ZERO config, no API key, instant result
# ---------------------------------------------------------------------------
@mcp.tool()
def quick_scan(description: str) -> dict:
    """Describe an AI system -> instant transparency and explainability assessment. No API key required."""
    limit_err = _check_rate_limit("quick_scan_anonymous")
    if limit_err:
        return {"error": "rate_limited", "message": limit_err}

    transparency_level, transparency_score, positive_signals = _assess_transparency_level(description)
    text_lower = description.lower()

    # Determine model type
    detected_type = "unknown"
    for mtype in MODEL_TYPES:
        if mtype in text_lower or (mtype == "nlp" and any(w in text_lower for w in ["language", "text", "chat", "llm"])):
            detected_type = mtype
            break
    if detected_type == "unknown" and any(w in text_lower for w in ["image", "vision", "photo", "video"]):
        detected_type = "computer_vision"
    if detected_type == "unknown" and any(w in text_lower for w in ["generat", "create", "synthes"]):
        detected_type = "generative"
    if detected_type == "unknown" and any(w in text_lower for w in ["recommend", "suggest", "personali"]):
        detected_type = "recommendation"
    if detected_type == "unknown" and any(w in text_lower for w in ["predict", "classif", "detect"]):
        detected_type = "classification"

    # Determine if high-risk (needs full transparency)
    high_risk_keywords = [
        "hiring", "recruit", "loan", "credit", "insurance", "medical", "diagnosis",
        "judicial", "law enforcement", "biometric", "education", "grading",
    ]
    is_high_risk = bool(_match_keywords(description, high_risk_keywords))

    # Recommend explainability methods
    recommended_methods = []  # type: List[str]
    if detected_type in MODEL_TYPES:
        method_keys = MODEL_TYPES[detected_type]["explainability_methods"]
        for mk in method_keys:
            if mk in EXPLAINABILITY_METHODS:
                recommended_methods.append(EXPLAINABILITY_METHODS[mk]["name"])

    # Build top actions
    if transparency_level == "low":
        top_actions = [
            "URGENT: Create model documentation covering purpose, capabilities, and limitations",
            "Implement at least one explainability method ({})".format(
                recommended_methods[0] if recommended_methods else "SHAP or LIME"
            ),
            "Document known limitations and failure modes for deployers",
        ]
    elif transparency_level == "moderate":
        top_actions = [
            "Strengthen documentation with quantitative performance metrics per group",
            "Add human-readable decision explanations for end users",
            "Conduct a transparency audit against EU AI Act Article 13",
        ]
    else:
        top_actions = [
            "Good transparency baseline -- formalise into EU AI Act Article 13 compliant documentation",
            "Consider generating a model card for public disclosure",
            "Implement ongoing transparency monitoring for model updates",
        ]

    return {
        "transparency_level": transparency_level,
        "transparency_score": transparency_score,
        "positive_signals": positive_signals,
        "detected_model_type": detected_type,
        "is_high_risk": is_high_risk,
        "recommended_explainability_methods": recommended_methods if recommended_methods else ["SHAP", "LIME", "Counterfactual Explanations"],
        "top_3_actions": top_actions,
        "eu_ai_act_relevance": (
            "HIGH-RISK: Article 13 transparency obligations are MANDATORY. "
            "Full technical documentation per Annex IV required."
            if is_high_risk
            else "Transparency obligations under Article 50 may apply (user disclosure, content labelling)."
        ),
        "next_step": "Use generate_model_card for structured documentation or transparency_audit for full assessment",
        "meok_labs": "https://meok.ai",
    }


# ---------------------------------------------------------------------------
# Tool: generate_model_card
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_model_card(
    model_name: str,
    purpose: str,
    training_data: str = "",
    api_key: str = "",
) -> dict:
    """Generate an EU AI Act compliant model card with structured transparency information.

    Args:
        model_name: Name of the AI model or system.
        purpose: Description of the model's intended purpose.
        training_data: Description of training data used (leave empty if not available).
        api_key: Optional MEOK API key for pro tier.
    """
    allowed, msg, tier = check_access(api_key)
    if not allowed:
        return {"error": msg, "upgrade_url": "https://meok.ai/pricing"}
    limit_err = _check_rate_limit("model_card", tier)
    if limit_err:
        return {"error": "rate_limited", "message": limit_err}

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Detect model type from purpose
    purpose_lower = purpose.lower()
    detected_type = "classification"
    for mtype in MODEL_TYPES:
        if mtype in purpose_lower:
            detected_type = mtype
            break
    if any(w in purpose_lower for w in ["language", "text", "chat", "llm", "nlp"]):
        detected_type = "nlp"
    elif any(w in purpose_lower for w in ["image", "vision", "photo"]):
        detected_type = "computer_vision"
    elif any(w in purpose_lower for w in ["generat", "create"]):
        detected_type = "generative"
    elif any(w in purpose_lower for w in ["recommend", "suggest"]):
        detected_type = "recommendation"

    type_info = MODEL_TYPES.get(detected_type, MODEL_TYPES["classification"])

    # Build model card
    model_card = """# Model Card: {model_name}

**Generated:** {date}
**Standard:** EU AI Act (Regulation (EU) 2024/1689) Article 13 + Mitchell et al. Model Cards framework
**Generator:** MEOK AI Labs Explainability Report Server (https://meok.ai)

---

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | {model_name} |
| **Model Version** | [Version number] |
| **Model Type** | {model_type} |
| **Date** | {date} |
| **Provider** | [Organisation name] |
| **Contact** | [Contact email] |
| **License** | [License type] |

## Intended Use

### Primary Intended Uses
{purpose}

### Primary Intended Users
[Describe the intended users of this model]

### Out-of-Scope Uses
[Describe use cases this model should NOT be used for]

---

## Training Data

{training_data_section}

### Data Governance (EU AI Act Article 10)
- [ ] Data governance practices documented
- [ ] Training data examined for biases (Article 10(2)(f))
- [ ] Data is representative of deployment population (Article 10(3))
- [ ] Personal data processing complies with GDPR

---

## Performance Metrics

### Recommended Metrics for {model_type}
{metrics_section}

### Disaggregated Evaluation
[Report performance broken down by relevant demographic groups]

| Group | {metric_headers} |
|-------|{metric_dashes}|
| [Group 1] | [Values] |
| [Group 2] | [Values] |
| [Overall] | [Values] |

---

## Limitations and Risks

### Known Limitations
[Document known limitations, failure modes, and edge cases]

### Ethical Considerations
[Document ethical considerations, potential harms, and mitigations]

### Bias Assessment
[Document known biases and mitigation measures applied]

---

## Explainability

### Recommended Explainability Methods
{explainability_section}

### Human Oversight (EU AI Act Article 14)
- [ ] Human oversight measures are identified and documented
- [ ] System enables overseers to understand capabilities and limitations
- [ ] Override or intervention mechanisms are available
- [ ] Automation bias risks are addressed

---

## EU AI Act Compliance (Article 13)

- [ ] Instructions for use accompany the system
- [ ] Provider identity and contact details included
- [ ] Capabilities and limitations described
- [ ] Intended purpose and preconditions specified
- [ ] Accuracy levels declared with appropriate metrics
- [ ] Known circumstances affecting performance documented
- [ ] Human oversight measures described
- [ ] Expected lifetime and maintenance requirements specified
- [ ] Computational and hardware requirements documented
- [ ] Input data specifications provided
- [ ] Output interpretation guidance provided

---

## Document Control

| Field | Value |
|-------|-------|
| Last Updated | {date} |
| Review Cycle | [Quarterly/Upon significant change] |
| Approved By | [Name and role] |

---

*Generated by MEOK AI Labs | https://meok.ai*
*This template follows EU AI Act Article 13 requirements and the Model Cards framework.*
""".format(
        model_name=model_name,
        date=date_str,
        model_type=detected_type,
        purpose=purpose,
        training_data_section=training_data if training_data else "[Describe the training data: sources, size, collection methodology, time period, demographic composition]",
        metrics_section="\n".join("- {}".format(m) for m in type_info["metrics"]),
        metric_headers=" | ".join(type_info["metrics"][:4]),
        metric_dashes=" | ".join(["---"] * min(4, len(type_info["metrics"]))),
        explainability_section="\n".join(
            "- **{}**: {}".format(
                EXPLAINABILITY_METHODS[mk]["name"],
                EXPLAINABILITY_METHODS[mk]["description"],
            )
            for mk in type_info["explainability_methods"]
            if mk in EXPLAINABILITY_METHODS
        ),
    )

    return {
        "format": "markdown",
        "model_card": model_card,
        "detected_model_type": detected_type,
        "recommended_metrics": type_info["metrics"],
        "recommended_explainability": [
            EXPLAINABILITY_METHODS[mk]["name"]
            for mk in type_info["explainability_methods"]
            if mk in EXPLAINABILITY_METHODS
        ],
        "sections_to_complete": [
            "Model Version",
            "Provider and Contact",
            "Primary Intended Users",
            "Out-of-Scope Uses",
            "Training Data details" if not training_data else None,
            "Performance Metrics values",
            "Disaggregated Evaluation results",
            "Known Limitations",
            "Ethical Considerations",
            "Bias Assessment",
        ],
        "eu_ai_act_note": "Article 13 requires this information for all high-risk AI systems before market placement.",
        "meok_labs": "https://meok.ai",
    }


# ---------------------------------------------------------------------------
# Tool: explain_decision
# ---------------------------------------------------------------------------
@mcp.tool()
def explain_decision(
    decision: str,
    factors: str = "",
    api_key: str = "",
) -> dict:
    """Generate a human-readable explanation of an AI decision.

    Args:
        decision: The AI decision or prediction to explain (e.g. "Loan application denied").
        factors: Comma-separated contributing factors (e.g. "credit_score:620,income:35000,debt_ratio:0.45"). Leave empty for generic guidance.
        api_key: Optional MEOK API key for pro tier.
    """
    allowed, msg, tier = check_access(api_key)
    if not allowed:
        return {"error": msg, "upgrade_url": "https://meok.ai/pricing"}
    limit_err = _check_rate_limit("explain_decision", tier)
    if limit_err:
        return {"error": "rate_limited", "message": limit_err}

    # Parse factors if provided
    parsed_factors = []  # type: List[Dict[str, str]]
    factor_explanations = []  # type: List[str]

    if factors:
        for factor_pair in factors.split(","):
            factor_pair = factor_pair.strip()
            if ":" in factor_pair:
                name, value = factor_pair.split(":", 1)
                name = name.strip()
                value = value.strip()
                parsed_factors.append({"factor": name, "value": value})

                # Generate human-readable explanation for common factor types
                name_lower = name.lower().replace("_", " ")
                try:
                    numeric_val = float(value)
                    factor_explanations.append(
                        "Your {} was {} -- this {} the decision.".format(
                            name_lower,
                            value,
                            "positively influenced" if numeric_val > 0.5 else "negatively influenced"
                        )
                    )
                except ValueError:
                    factor_explanations.append(
                        "Your {} was '{}' -- this was a factor in the decision.".format(name_lower, value)
                    )
            else:
                parsed_factors.append({"factor": factor_pair, "value": "present"})
                factor_explanations.append("{} was a contributing factor.".format(factor_pair))

    # Sort factors by assumed importance (simple heuristic: length of value string as proxy)
    if parsed_factors:
        # Assume factors are listed in order of importance
        primary_factors = parsed_factors[:3]
        secondary_factors = parsed_factors[3:]
    else:
        primary_factors = []
        secondary_factors = []

    # Generate explanation
    explanation_parts = []  # type: List[str]

    # Opening
    explanation_parts.append("Decision: {}".format(decision))
    explanation_parts.append("")

    if parsed_factors:
        explanation_parts.append("This decision was based on the following factors:")
        explanation_parts.append("")

        if primary_factors:
            explanation_parts.append("Primary factors:")
            for i, f in enumerate(primary_factors, 1):
                explanation_parts.append("  {}. {} = {}".format(i, f["factor"], f["value"]))

        if secondary_factors:
            explanation_parts.append("")
            explanation_parts.append("Additional factors:")
            for f in secondary_factors:
                explanation_parts.append("  - {} = {}".format(f["factor"], f["value"]))
    else:
        explanation_parts.append(
            "No specific factors were provided. To generate a detailed explanation, "
            "provide factors as comma-separated name:value pairs."
        )

    # Determine if decision involves protected attributes
    decision_lower = decision.lower()
    sensitive_decision = any(
        kw in decision_lower
        for kw in ["denied", "rejected", "declined", "terminated", "suspended", "refused", "flagged"]
    )

    # Counterfactual suggestion
    counterfactual = None
    if parsed_factors and sensitive_decision:
        # Suggest what would need to change
        counterfactual_parts = []  # type: List[str]
        for f in primary_factors:
            try:
                val = float(f["value"])
                # Suggest improvement direction
                counterfactual_parts.append(
                    "If {} were {} (instead of {}), the outcome might differ.".format(
                        f["factor"],
                        round(val * 1.2, 2) if val < 1 else round(val * 1.1, 0),
                        f["value"],
                    )
                )
            except ValueError:
                pass
        if counterfactual_parts:
            counterfactual = " ".join(counterfactual_parts)

    return {
        "decision": decision,
        "explanation": "\n".join(explanation_parts),
        "factors_analyzed": parsed_factors,
        "factor_explanations": factor_explanations if factor_explanations else ["Provide factors for detailed explanations"],
        "primary_factors": primary_factors,
        "secondary_factors": secondary_factors,
        "counterfactual_suggestion": counterfactual,
        "is_adverse_decision": sensitive_decision,
        "right_to_explanation": (
            "Under GDPR Article 22 and EU AI Act Article 86, individuals have the right "
            "to obtain meaningful information about the logic involved in automated decisions "
            "that significantly affect them."
            if sensitive_decision
            else "Standard transparency obligations apply."
        ),
        "recommended_explainability_methods": [
            "SHAP (SHapley Additive exPlanations) -- quantifies each feature's contribution",
            "LIME (Local Interpretable Model-agnostic Explanations) -- local linear approximation",
            "Counterfactual Explanations -- what would need to change for a different outcome",
        ],
        "eu_ai_act_articles": [
            "Article 13 -- transparency and provision of information to deployers",
            "Article 86 -- right to explanation of individual decision-making",
            "Article 14 -- human oversight of AI decisions",
        ],
        "meok_labs": "https://meok.ai",
    }


# ---------------------------------------------------------------------------
# Tool: transparency_audit
# ---------------------------------------------------------------------------
@mcp.tool()
def transparency_audit(
    system_description: str,
    api_key: str = "",
) -> dict:
    """Assess an AI system against EU AI Act Article 13 transparency requirements.

    Args:
        system_description: Detailed description of the AI system including its purpose, capabilities, data, and deployment context.
        api_key: Optional MEOK API key for pro tier.
    """
    allowed, msg, tier = check_access(api_key)
    if not allowed:
        return {"error": msg, "upgrade_url": "https://meok.ai/pricing"}
    limit_err = _check_rate_limit("transparency_audit", tier)
    if limit_err:
        return {"error": "rate_limited", "message": limit_err}

    text_lower = system_description.lower()

    # Check each Article 13 requirement
    article_13_results = []  # type: List[Dict[str, object]]
    passed = 0
    total = 0

    keyword_map = {
        "T1": ["instruction", "guide", "manual", "documentation", "how to use"],
        "T2": ["provider", "developer", "company", "organisation", "contact", "email"],
        "T3": ["capabilit", "limitation", "can do", "cannot", "performance", "what it does"],
        "T4": ["purpose", "intended", "designed for", "use case", "objective"],
        "T5": ["accuracy", "precision", "recall", "f1", "performance metric", "auc"],
        "T6": ["limitation", "failure", "edge case", "does not work", "caveat", "known issue"],
        "T7": ["human oversight", "human review", "human-in-the-loop", "override", "intervention"],
        "T8": ["lifetime", "maintenance", "update", "version", "deprecat"],
        "T9": ["hardware", "compute", "gpu", "cpu", "memory", "resource", "infrastructure"],
        "T10": ["input", "data format", "data type", "specification", "schema", "api"],
        "T11": ["interpret", "explain", "output meaning", "confidence", "probability", "score meaning"],
        "T12": ["changelog", "version history", "update log", "modification"],
    }

    for check in TRANSPARENCY_REQUIREMENTS["article_13"]["checks"]:
        total += 1
        keywords = keyword_map.get(check["id"], [])
        matched = _match_keywords(system_description, keywords)
        status = "LIKELY_PRESENT" if matched else "NOT_DETECTED"
        if matched:
            passed += 1

        article_13_results.append({
            "id": check["id"],
            "requirement": check["requirement"],
            "category": check["category"],
            "status": status,
            "evidence": matched if matched else ["No matching keywords found in description"],
        })

    # Check Article 50 requirements
    article_50_results = []  # type: List[Dict[str, object]]
    a50_keywords = {
        "A50-1": ["inform user", "ai disclosure", "chatbot", "bot notice", "ai system notice"],
        "A50-2": ["generated content", "ai generated", "synthetic", "artificially generated", "machine generated"],
        "A50-3": ["emotion recognition", "biometric", "affect detection"],
        "A50-4": ["deepfake", "deep fake", "face swap", "synthetic media"],
    }

    for check in TRANSPARENCY_REQUIREMENTS["article_50"]["checks"]:
        keywords = a50_keywords.get(check["id"], [])
        matched = _match_keywords(system_description, keywords)
        applicable = bool(matched) or any(kw in text_lower for kw in keywords)
        article_50_results.append({
            "id": check["id"],
            "requirement": check["requirement"],
            "applicable": applicable,
            "status": "LIKELY_PRESENT" if matched else ("POTENTIALLY_APPLICABLE" if applicable else "NOT_APPLICABLE"),
        })

    # Overall score
    score = (passed / total * 100) if total > 0 else 0

    if score >= 70:
        assessment = "GOOD -- strong transparency foundation, minor gaps to address"
    elif score >= 40:
        assessment = "MODERATE -- several transparency requirements need attention"
    else:
        assessment = "POOR -- significant transparency gaps; major work needed for Article 13 compliance"

    # Priority actions
    missing = [r for r in article_13_results if r["status"] == "NOT_DETECTED"]
    priority_actions = []  # type: List[str]
    for m in missing[:5]:
        priority_actions.append("Address: {} ({})".format(m["requirement"], m["id"]))

    return {
        "transparency_score": "{:.1f}%".format(score),
        "assessment": assessment,
        "article_13_checklist": article_13_results,
        "article_50_checklist": article_50_results,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "gaps": total - passed,
        },
        "priority_actions": priority_actions,
        "recommended_tools": [
            "generate_model_card -- create structured transparency documentation",
            "explain_decision -- generate human-readable decision explanations",
            "create_impact_assessment -- assess impact on affected groups",
        ],
        "eu_ai_act_note": (
            "Article 13 requires high-risk AI systems to be sufficiently transparent "
            "for deployers to interpret output and use appropriately. Non-compliance "
            "may result in penalties up to EUR 15,000,000 or 3% of global turnover."
        ),
        "meok_labs": "https://meok.ai",
    }


# ---------------------------------------------------------------------------
# Tool: create_impact_assessment
# ---------------------------------------------------------------------------
@mcp.tool()
def create_impact_assessment(
    system_name: str,
    affected_groups: str = "",
    api_key: str = "",
) -> dict:
    """Generate a DPIA/AIIA (AI Impact Assessment) template for an AI system.

    Args:
        system_name: Name of the AI system.
        affected_groups: Comma-separated list of affected groups (e.g. "employees,customers,public"). Leave empty for generic template.
        api_key: Optional MEOK API key for pro tier.
    """
    allowed, msg, tier = check_access(api_key)
    if not allowed:
        return {"error": msg, "upgrade_url": "https://meok.ai/pricing"}
    limit_err = _check_rate_limit("impact_assessment", tier)
    if limit_err:
        return {"error": "rate_limited", "message": limit_err}

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Parse affected groups
    groups = []  # type: List[str]
    if affected_groups:
        groups = [g.strip() for g in affected_groups.split(",") if g.strip()]
    else:
        groups = ["Direct users", "Affected individuals", "Third parties", "General public"]

    # Build impact matrix
    impact_matrix = []  # type: List[Dict[str, object]]
    for category_key, category_info in RISK_CATEGORIES.items():
        category_impacts = []  # type: List[Dict[str, str]]
        for area in category_info["areas"]:
            group_impacts = {}  # type: Dict[str, str]
            for group in groups:
                group_impacts[group] = "[Assess: None / Low / Medium / High / Critical]"
            category_impacts.append({
                "risk_area": area,
                "group_impacts": group_impacts,
                "mitigation": "[Describe mitigation measures]",
            })
        impact_matrix.append({
            "category": category_info["name"],
            "key": category_key,
            "impacts": category_impacts,
        })

    # Generate template
    template = """# AI Impact Assessment (AIIA)
## {system_name}

**Assessment Date:** {date}
**Assessor:** [Name and role]
**Status:** [Draft / Under Review / Approved]
**Regulation:** EU AI Act (Regulation (EU) 2024/1689) Articles 9, 27
**Generated by:** MEOK AI Labs (https://meok.ai)

---

## 1. System Overview

| Field | Value |
|-------|-------|
| System Name | {system_name} |
| Provider | [Organisation name] |
| Version | [Version number] |
| Deployment Date | [Planned/actual date] |
| Risk Classification | [Prohibited / High-Risk / Limited / Minimal] |

### Purpose and Scope
[Describe the system's purpose, scope of deployment, and scale of impact]

### Affected Groups
{affected_groups_section}

---

## 2. Fundamental Rights Impact Assessment (EU AI Act Article 27)

For each risk area, assess the impact on each affected group.

### 2.1 Fundamental Rights
{fundamental_rights_section}

### 2.2 Safety Impact
{safety_section}

### 2.3 Societal Impact
{societal_section}

---

## 3. Data Protection Impact Assessment (GDPR Article 35)

- [ ] Processing involves systematic and extensive profiling
- [ ] Processing involves large-scale use of sensitive data
- [ ] Processing involves systematic monitoring of publicly accessible areas
- [ ] A DPIA is required under GDPR Article 35

### Data Processing Activities
[List all personal data processing activities]

### Legal Basis
[Document the legal basis for processing under GDPR Article 6]

### Data Subject Rights
[Document how data subject rights are facilitated]

---

## 4. Risk Assessment and Mitigation

### Identified Risks
| Risk | Likelihood | Impact | Risk Level | Mitigation |
|------|-----------|--------|------------|------------|
| [Risk 1] | [Low/Med/High] | [Low/Med/High] | [Score] | [Mitigation] |
| [Risk 2] | [Low/Med/High] | [Low/Med/High] | [Score] | [Mitigation] |

### Residual Risks
[Document residual risks after mitigation measures are applied]

---

## 5. Stakeholder Consultation

### Consulted Parties
- [ ] Affected individuals or their representatives
- [ ] Data Protection Officer
- [ ] Legal counsel
- [ ] Domain experts
- [ ] Civil society organisations (where appropriate)

### Consultation Outcomes
[Document feedback received and how it was incorporated]

---

## 6. Monitoring and Review

### Monitoring Plan
- [ ] Regular performance monitoring established
- [ ] Bias monitoring in place
- [ ] Incident reporting procedure defined
- [ ] Review schedule set ([quarterly/annually])

### Review Triggers
- Material change in system capabilities
- Expansion to new affected groups
- Incident or complaint
- Regulatory update

---

## 7. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Assessor | | | |
| DPO | | | |
| Legal | | | |
| Management | | | |

---

*Generated by MEOK AI Labs | https://meok.ai*
*This template follows EU AI Act Article 27 (Fundamental Rights Impact Assessment)*
*and GDPR Article 35 (Data Protection Impact Assessment) requirements.*
""".format(
        system_name=system_name,
        date=date_str,
        affected_groups_section="\n".join("- {}".format(g) for g in groups),
        fundamental_rights_section="\n".join(
            "- {}: [Assess impact]".format(area)
            for area in RISK_CATEGORIES["fundamental_rights"]["areas"]
        ),
        safety_section="\n".join(
            "- {}: [Assess impact]".format(area)
            for area in RISK_CATEGORIES["safety"]["areas"]
        ),
        societal_section="\n".join(
            "- {}: [Assess impact]".format(area)
            for area in RISK_CATEGORIES["societal"]["areas"]
        ),
    )

    return {
        "format": "markdown",
        "template": template,
        "system_name": system_name,
        "affected_groups": groups,
        "impact_categories": [
            {"category": "Fundamental Rights", "areas": len(RISK_CATEGORIES["fundamental_rights"]["areas"])},
            {"category": "Safety", "areas": len(RISK_CATEGORIES["safety"]["areas"])},
            {"category": "Societal", "areas": len(RISK_CATEGORIES["societal"]["areas"])},
        ],
        "total_risk_areas": sum(len(c["areas"]) for c in RISK_CATEGORIES.values()),
        "sections_to_complete": [
            "System Overview",
            "Impact assessments for each risk area and group",
            "Data Protection Impact Assessment",
            "Risk Assessment and Mitigation",
            "Stakeholder Consultation",
            "Monitoring and Review plan",
            "Approval signatures",
        ],
        "eu_ai_act_articles": [
            "Article 9 -- Risk Management System",
            "Article 27 -- Fundamental Rights Impact Assessment (mandatory for high-risk deployers)",
            "GDPR Article 35 -- Data Protection Impact Assessment",
        ],
        "meok_labs": "https://meok.ai",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the explainability-report-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
