import pandas as pd
import matplotlib.pyplot as plt


# Model performance data

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],

    "Accuracy": [
        0.7532,
        0.7468,
        0.7662
    ],

    "AUC": [
        0.8228,
        0.8340,
        0.8332
    ],

    "F1-score": [
        0.64,
        0.65,
        0.65
    ],

    "Recall": [
        0.62,
        0.67,
        0.62
    ]

})


print(results)


# Save table

results.to_csv(
    "results/model_comparison.csv",
    index=False
)


# Accuracy comparison

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.ylabel("Accuracy")

plt.title(
    "Model Accuracy Comparison"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.show()



# AUC comparison

plt.figure(figsize=(8,5))

plt.bar(
    results["Model"],
    results["AUC"]
)

plt.ylabel("AUC")

plt.title(
    "Model AUC Comparison"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "results/AUC_comparison.png",
    dpi=300
)

plt.show()