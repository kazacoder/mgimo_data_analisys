# ===============================
# Аналитика для бизнеса — итоговое задание
# Использование готового датасета
# Тема: Повышение точности подготовки Паспорта проекта
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------
# 1. Загрузка готового датасета
# -------------------------------------------------------

# Путь к датасету
DATA_PATH = "./passport_projects_dataset.csv"
assert os.path.exists(DATA_PATH), f"Файл {DATA_PATH} не найден"

df = pd.read_csv(DATA_PATH)

# Просмотр структуры
print("=== Структура датасета ===")
print(df.head())

# -------------------------------------------------------
# 2. Проверка и базовая подготовка данных
# -------------------------------------------------------

# Приведение числовых колонок к числовому типу (на случай текстовых остатков)
num_cols = ["age_years", "weight_kg", "est_total_revenue", "processing_cost",
            "break_even", "relative_margin_pct"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Удалим явные выбросы и пропуски
df = df[df["weight_kg"] > 0]
df = df.dropna(subset=["est_total_revenue", "processing_cost"])

# -------------------------------------------------------
# 3. Расчёт агрегированных показателей
# -------------------------------------------------------

summary = {
    "Всего партий": len(df),
    "Доля убыточных (break_even < 0)": f"{(df['break_even'] < 0).mean() * 100:.1f}%",
    "Средний break_even": f"{df['break_even'].mean():,.0f} ₽",
    "Средняя выручка от переработки": f"{df['est_total_revenue'].mean():,.0f} ₽",
    "Средняя стоимость переработки": f"{df['processing_cost'].mean():,.0f} ₽",
    "Средняя относительная маржа (%)": f"{df['relative_margin_pct'].mean():.2f}%"
}

print("\n=== Ключевые показатели ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# -------------------------------------------------------
# 4. Визуализации
# -------------------------------------------------------

os.makedirs("report_assets", exist_ok=True)
plt.style.use("default")

# Scatter: выручка vs стоимость переработки
plt.figure(figsize=(8,6))
sns.scatterplot(x="processing_cost", y="est_total_revenue", data=df, hue="type", alpha=0.7)
plt.plot([0, df["est_total_revenue"].max()], [0, df["est_total_revenue"].max()], 'r--', label='y = x')
plt.xlabel("Стоимость переработки, ₽")
plt.ylabel("Оценочная выручка, ₽")
plt.title("Соотношение выручки и стоимости переработки")
plt.legend()
plt.tight_layout()
plt.savefig("report_assets/scatter_cost_vs_revenue.png")
plt.show()
plt.close()

# Boxplot: выручка по типам техники
plt.figure(figsize=(8,6))
sns.boxplot(x="type", y="est_total_revenue", data=df)
plt.xticks(rotation=30)
plt.ylabel("Выручка, ₽")
plt.title("Распределение выручки по типам техники")
plt.tight_layout()
plt.savefig("report_assets/box_revenue_by_type.png")
plt.show()
plt.close()

# Histogram: распределение break_even
plt.figure(figsize=(8,6))
sns.histplot(df["break_even"], bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--", label="Порог безубыточности")
plt.xlabel("Break-even (руб.)")
plt.title("Распределение порога безубыточности (break_even)")
plt.legend()
plt.tight_layout()
plt.savefig("report_assets/hist_break_even.png")
plt.show()
plt.close()

# Bar chart: % убыточных лотов по типам
loss_ratio = df.groupby("type")["break_even"].apply(lambda x: (x < 0).mean() * 100).sort_values()
plt.figure(figsize=(8,6))
sns.barplot(x=loss_ratio.index, y=loss_ratio.values)
plt.xticks(rotation=30)
plt.ylabel("% убыточных партий")
plt.title("Доля убыточных партий по типам техники")
plt.tight_layout()
plt.savefig("report_assets/bar_unprof_by_type.png")
plt.show()
plt.close()

# Heatmap: корреляционная матрица
plt.figure(figsize=(8,6))
corr_cols = ["age_years", "weight_kg", "est_total_revenue", "processing_cost", "break_even"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.tight_layout()
plt.savefig("report_assets/heatmap_correlation.png")
plt.show()
plt.close()

# -------------------------------------------------------
# 5. Вывод результатов
# -------------------------------------------------------

print("\n✅ Анализ завершён.")
print("Графики сохранены в папку report_assets/")
print(f"Всего графиков: 5")