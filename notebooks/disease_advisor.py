import json
import os
from textwrap import fill

class DiseaseAdvisor:
    def __init__(self, json_path=None):
        """
        Load the disease knowledge base JSON file safely.
        """
        if json_path is None:
            # Auto-detect absolute path to your knowledge file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, "../knowledge/disease_knowledge_base.json")

        json_path = os.path.normpath(json_path)  # clean path for Windows compatibility

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"⚠️ Knowledge base not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

        print(f"✅ Loaded knowledge base with {len(self.kb)} diseases from {json_path}")

    def get_info(self, predicted_label):
        """Fetch disease details and recommendations for a predicted label."""
        if predicted_label not in self.kb:
            return {
                "error": f"❌ '{predicted_label}' not found in knowledge base.",
                "suggestion": "Check label formatting (e.g. underscores, spelling)."
            }

        disease_data = self.kb[predicted_label]
        name_clean = predicted_label.replace("___", " ").replace("_", " ")

        return {
            "disease": name_clean,
            "description": disease_data["description"],
            "recommendations": disease_data["recommendation"]
        }

    def pretty_print(self, predicted_label):
        """Print formatted, user-friendly disease information."""
        info = self.get_info(predicted_label)
        if "error" in info:
            print(info["error"])
            if "suggestion" in info:
                print("💡", info["suggestion"])
            return

        print("=" * 60)
        print(f"🌿 Disease: {info['disease']}")
        print("-" * 60)
        print(f"🧠 Description:\n{fill(info['description'], width=70)}")
        print("\n💡 Recommendations:")
        for i, rec in enumerate(info["recommendations"], 1):
            print(f"  {i}. {rec}")
        print("=" * 60)


# === Example Usage ===
if __name__ == "__main__":
    # No need to manually enter full path unless you want to override it
    advisor = DiseaseAdvisor()

    # Example prediction output from your model
    sample_prediction = "Tomato___Early_blight"
    print(f"\n🔍 Querying knowledge base for: {sample_prediction}")
    advisor.pretty_print(sample_prediction)
