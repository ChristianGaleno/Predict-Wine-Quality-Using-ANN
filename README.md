# 🍷 Wine Quality Dataset — Cleaned & Processed

## 📌 Description
This dataset contains physicochemical tests of red and white wines, along with their quality scores.

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 📊 Dataset Summary
- **Total Rows:** 6,497
- **Total Columns:** 12
- **Target Variable:** `quality` (integer values from 3 to 9)
- **File Format:** CSV (UTF-8)
- **Missing Values:** None
- **Duplicates Removed:** Yes

---

## 📂 Column Description

| Column Name            | Data Type | Description |
|------------------------|-----------|-------------|
| fixed acidity          | float     | g(tartaric acid)/dm³ |
| volatile acidity       | float     | g(acetic acid)/dm³ |
| citric acid            | float     | g/dm³ |
| residual sugar         | float     | g/dm³ |
| chlorides              | float     | g(sodium chloride)/dm³ |
| free sulfur dioxide    | float     | mg/dm³ |
| total sulfur dioxide   | float     | mg/dm³ |
| density                | float     | g/cm³ |
| pH                     | float     | Acidity level |
| sulphates              | float     | g(potassium sulphate)/dm³ |
| alcohol                | float     | % by volume |
| quality                | int       | Quality score (3-9) |

---

## 🎯 Possible Use Cases
- **Classification:** Predict wine quality categories (e.g., Low, Medium, High)
- **Regression:** Predict numeric wine quality scores
- **Data Visualization:** Explore relationships between chemical properties and quality
- **Feature Engineering:** Create new derived features for ML models
- **Teaching Dataset:** Perfect for EDA & beginner ML projects

---

## 📈 Example Insights
- Higher **alcohol** content generally correlates with better quality wines.
- **Volatile acidity** has a negative correlation with quality.
- Red and white wines show different distributions in acidity and sulphates.

---

## ⚠️ License
**CC BY 4.0** — You are free to share and adapt this dataset for any purpose, with attribution.

---

## 📝 Acknowledgements
- Original dataset by Paulo Cortez et al., University of Minho, Portugal.
