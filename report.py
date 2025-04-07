import torch
from models.model import DISEASE_LABELS

def get_predictions(outputs, threshold=0.5):

    probs = torch.sigmoid(outputs)
    predictions = (probs > threshold).float()
    return predictions, probs

def format_findings(predictions, probabilities, labels=DISEASE_LABELS, threshold=0.5):
    findings = []
    #pairing each probability with its disease labels
    prob_label_pairs = [(prob.item(), labels[i]) for i, prob in enumerate(probabilities[0])]
    # sorting them according but we didnt use it 
    prob_label_pairs.sort(reverse=True)
    
    top_prob, top_label = prob_label_pairs[0]
    if top_prob >= threshold:
        confidence = f"{top_prob*100:.1f}%"
        findings.append(f"{top_label} (Confidence: {confidence})")
    else:
        findings.append("No significant abnormalities detected")
    
    return findings

def generate_report(findings, visual_features):
    """Structure the findings into a formatted report (optional local generation)."""
    report = {
        "findings": findings,
        "visual_features": visual_features,
        "impression": [],
        "recommendations": []
    }
    
    severity_levels = {
        "critical": ["Pneumonia", "Pneumothorax"],
        "moderate": ["Cardiomegaly", "Effusion", "Mass"],
        "mild": ["Nodule", "Atelectasis"]
    }
    # check the worst one and also try to work on which one needs more attentions
    max_severity = "normal"
    for finding in findings:
        for severity, conditions in severity_levels.items():
            if any(condition in finding for condition in conditions):
                if severity == "critical":
                    max_severity = severity
                    break
                elif severity == "moderate" and max_severity != "critical":
                    max_severity = severity
                elif severity == "mild" and max_severity not in ["critical", "moderate"]:
                    max_severity = severity
    
    return {
        "severity": max_severity,
        "findings": findings,
        "visual_features": visual_features
    }

"""Severity Level	    Example Diseases	       Meaning
1. Critical --------	Pneumonia, Pneumothorax	----Life-threatening, needs urgent attention
2. Moderate---------	Cardiomegaly, Effusion	----Serious but not immediately fatal
3. Mild-------------	Nodule, Atelectasis	--------Less dangerous, may need monitoring
"""