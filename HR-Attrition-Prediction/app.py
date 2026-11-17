

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 1. إعدادات واجهة الصفحة
st.set_page_config(
    page_title="نظام التنبؤ باستقالة الموظفين", page_icon="📊", layout="centered"
)
st.title("📊 نظام التنبؤ الذكي بمغادرة الموظفين (HR Attrition)")
st.write(
    "أدخل بيانات الموظف أدناه ليقوم نموذج الذكاء الاصطناعي بحساب احتمالية"
    " استمراريته أو تركه للعمل."
)


# 2. تحميل البيانات وتدريب النموذج (مع استخدام الكاش لتسريع الأداء)
@st.cache_resource
def train_model():
    df = pd.read_csv("HRDataset_v14-selected-columns.csv")
    features = [
        "Salary",
        "PerfScoreID",
        "EmpSatisfaction",
        "DaysLateLast30",
        "MarriedID",
        "GenderID",
    ]
    X = df[features]
    y = df["Termd"]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
    )
    model.fit(X, y)
    return model


rf_model = train_model()

st.divider()

# 3. واجهة إدخال بيانات الموظف
col1, col2 = st.columns(2)

with col1:
    salary = st.number_input(
        "الراتب السنوي (Salary):",
        min_value=30000,
        max_value=250000,
        value=65000,
        step=1000,
    )
    satisfaction = st.slider(
        "الرضا الوظيفي (EmpSatisfaction):", min_value=1, max_value=5, value=3
    )
    days_late = st.number_input(
        "أيام التأخير آخر 30 يوماً (DaysLateLast30):",
        min_value=0,
        max_value=10,
        value=0,
    )

with col2:
    perf_score = st.selectbox(
        "تقييم الأداء (PerfScoreID):",
        options=[1, 2, 3, 4],
        index=2,
        help="1: يحتاج تحسين، 3: يفي بالتوقعات، 4: يتجاوز التوقعات",
    )
    married = st.selectbox(
        "الحالة الاجتماعية (MarriedID):",
        options=[0, 1],
        format_func=lambda x: "متزوج (1)" if x == 1 else "أعزب (0)",
    )
    gender = st.selectbox(
        "الجنس (GenderID):",
        options=[0, 1],
        format_func=lambda x: "ذكر (1)" if x == 1 else "أنثى (0)",
    )

st.divider()

# 4. زر التوقع وعرض النتيجة
if st.button("🔍 تحليل وتوقع حالة الموظف", use_container_width=True):
    # تجهيز المدخلات بنفس ترتيب تدريب النموذج
    input_data = pd.DataFrame(
        [[salary, perf_score, satisfaction, days_late, married, gender]],
        columns=[
            "Salary",
            "PerfScoreID",
            "EmpSatisfaction",
            "DaysLateLast30",
            "MarriedID",
            "GenderID",
        ],
    )

    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][
        1
    ]  # احتمالية المغادرة (الفئة 1)

    if prediction == 1:
        st.error(
            "⚠️ **تنبيه مرتفع:** الموظف معرض لخطر مغادرة العمل بنسبة"
            f" **{probability * 100:.1f}%**"
        )
        st.info(
            "💡 **توصية:** يُنصح بمراجعة الراتب أو عقد جلسة تقييم للرضا الوظيفي"
            " مع الموظف."
        )
    else:
        st.success(
            "✅ **وضع مستقر:** من المرجح استمرار الموظف في عمله (احتمالية"
            f" المغادرة: **{probability * 100:.1f}%**)"
        )