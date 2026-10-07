import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. إعداد واجهة المستخدم والأيقونات
st.set_page_config(page_title="نظام الري الذكي بالذكاء الاصطناعي", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌱 نظام التحكم بالري الذكي (AI Smart Irrigation)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>هذا النظام يتنبأ بحاجة الحقل للري بناءً على قراءات مستشعرات التربة والطقس.</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. تحميل نموذج الذكاء الاصطناعي والمقياس المحفوظين من Jupyter
try:
    with open('irrigation_model.pkl', 'rb') as f:
        saved_objects = pickle.load(f)
        model = saved_objects['model']
        scaler = saved_objects['scaler']
        
    # 3. تصميم شريط المدخلات الجانبي (المستشعرات الخاصة بملفك المدمج)
    st.sidebar.header("📊 قراءات المستشعرات الحالية")
    temp = st.sidebar.slider("🌡️ درجة الحرارة (°C)", 15.0, 50.0, 25.0, 0.1)
    humidity = st.sidebar.slider("💧 رطوبة الجو (%)", 10.0, 100.0, 50.0, 0.1)
    soil_moisture = st.sidebar.slider("🪵 رطوبة التربة (%)", 10.0, 100.0, 40.0, 0.1)

    # 4. زر اتخاذ القرار والتنبؤ في الصفحة الرئيسية
    st.subheader("🔮 فحص وحساب قرار الري:")
    if st.button("🤖 اطلب قرار الذكاء الاصطناعي", type="primary", use_container_width=True):
        # موازنة وتعديل نطاق المدخلات الجديدة باستخدام الـ scaler
        input_data = np.array([[temp, humidity, soil_moisture]])
        input_scaled = scaler.transform(input_data)
        
        # إجراء التنبؤ وحساب الاحتمالية
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("📢 النتيجة واتخاذ القرار:")
        
        if prediction == 1:
            st.error(f"🚨 **قرار النظام: تشغيل مضخة الري فوراً!** (نسبة اليقين: {probability[1]*100:.1f}%)")
            st.info("💡 **التفسير:** رطوبة التربة منخفضة جداً والظروف الجوية تؤدي لجفاف الحقل، الموديل ينصح بالسقاية.")
        else:
            st.success(f"✅ **قرار النظام: إيقاف الري، لا داعي له حالياً.** (نسبة اليقين: {probability[0]*100:.1f}%)")
            st.info("💡 **التفسير:** التربة تمتلك رطوبة كافية مستقرة ولا توجد مؤشرات جفاف حالية.")
            
except FileNotFoundError:
    st.error("❌ خطأ: لم يتم العثور على ملف النموذج 'irrigation_model.pkl'. يرجى التأكد من تشغيل خلية الحفظ في Jupyter أولاً.")
2