
import sys
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from shap_utils import load_shap_explainers, explain_customer


# =========================================================
# 1. PROJECT SETUP
# =========================================================

st.set_page_config(
    page_title="Persona | Your Financial Homepage",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from decision_engine import load_models, add_propensity_scores, PRODUCTS
from homepage_personalizer import personalize_customer


# =========================================================
# 2. DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: #F5F7FB;
    color: #17233C;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #17233C !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #64748B;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E6EAF1;
    padding: 15px;
    border-radius: 13px;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
    background: white;
}

.stButton > button {
    border-radius: 9px;
}

.brand-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0 18px 0;
    border-bottom: 1px solid #E4E9F1;
    margin-bottom: 24px;
}

.brand-name {
    font-size: 25px;
    font-weight: 800;
    color: #18345C;
    letter-spacing: -1px;
}

.brand-caption {
    color: #718096;
    font-size: 12px;
    letter-spacing: 1.5px;
}

.welcome-title {
    font-size: 30px;
    font-weight: 750;
    color: #17233C;
    margin-bottom: 4px;
}

.welcome-subtitle {
    font-size: 15px;
    color: #718096;
    margin-bottom: 24px;
}

.offer-card {
    background: linear-gradient(120deg, #17345C, #315B91);
    border-radius: 18px;
    padding: 28px;
    color: white;
    min-height: 230px;
    margin: 10px 0 20px 0;
}

.offer-eyebrow {
    color: #C7D9F2;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

.offer-title {
    color: white;
    font-size: 29px;
    font-weight: 750;
    margin: 13px 0 8px 0;
}

.offer-description {
    color: #E0EAF8;
    font-size: 14px;
    max-width: 600px;
    line-height: 1.6;
}

.offer-note {
    color: #C7D9F2;
    font-size: 11px;
    margin-top: 18px;
}

.content-card {
    background: white;
    border: 1px solid #E6EAF1;
    border-radius: 15px;
    padding: 20px;
    min-height: 145px;
    margin-bottom: 12px;
}

.content-card-title {
    font-size: 17px;
    font-weight: 700;
    color: #1D3557;
    margin: 8px 0;
}

.content-card-description {
    font-size: 13px;
    color: #718096;
    line-height: 1.5;
}

.section-heading {
    font-size: 19px;
    font-weight: 700;
    color: #17233C;
    margin: 25px 0 12px 0;
}

.banner-card {
    background: #EAF0FF;
    border: 1px solid #D8E3FF;
    border-radius: 13px;
    padding: 17px 20px;
    color: #254477;
    margin: 12px 0 20px 0;
}

.muted-caption {
    color: #8793A5;
    font-size: 12px;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #E6EAF1;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODELS AND CUSTOMER DATA
# =========================================================

@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_customers():
    data_file = ROOT / "data" / "test.parquet"

    if not data_file.exists():
        raise FileNotFoundError(
            "test.parquet not found. Run src/split_data.py first."
        )

    df = pd.read_parquet(data_file)
    models = get_models()

    return add_propensity_scores(df, models)


try:
    models = get_models()
    customers = get_customers()

except Exception as error:
    st.error(f"Unable to load the application: {error}")
    st.stop()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def safe_value(value, default="Not available"):
    if pd.isna(value):
        return default
    return str(value)


def safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def render_content_card(title, description, eyebrow="FOR YOU"):
    st.markdown(f"""
    <div class="content-card">
        <div class="offer-eyebrow" style="color:#718096;">
            {html.escape(eyebrow)}
        </div>
        <div class="content-card-title">
            {html.escape(str(title))}
        </div>
        <div class="content-card-description">
            {html.escape(str(description))}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_offer(title, description, note):
    st.markdown(f"""
    <div class="offer-card">
        <div class="offer-eyebrow">Recommended for you</div>
        <div class="offer-title">{html.escape(str(title))}</div>
        <div class="offer-description">
            {html.escape(str(description))}
        </div>
        <div class="offer-note">{html.escape(str(note))}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 5. BRAND HEADER
# =========================================================

st.markdown("""
<div class="brand-row">
    <div>
        <div class="brand-name">◈ PERSONA</div>
        <div class="brand-caption">YOUR FINANCIAL EXPERIENCE</div>
    </div>
    <div class="brand-caption">PERSONALIZED HOMEPAGE</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. CUSTOMER SELECTOR
# =========================================================

st.markdown("### Preview a customer's homepage")

customer_ids = sorted(
    customers["customer_id"].astype(str).unique().tolist()
)

selected_id = st.selectbox(
    "Select customer",
    options=customer_ids,
    index=0,
    help="Choose a synthetic test customer to see their personalized homepage."
)

customer = customers[
    customers["customer_id"].astype(str) == selected_id
].iloc[0]

# The personalization engine determines the homepage content.
homepage = personalize_customer(customer, models)


# =========================================================
# 7. CUSTOMER-FACING HOMEPAGE
# =========================================================

customer_name = safe_value(
    customer.get("customer_name", None),
    "there"
)

st.markdown(
    f'<div class="welcome-title">Welcome back, {html.escape(customer_name)}!</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="welcome-subtitle">'
    'Here are some financial products and content selected for you.'
    '</div>',
    unsafe_allow_html=True
)


# Primary personalized recommendation
primary = safe_value(homepage.get("primary_slot"), "Explore our products")
secondary = safe_value(homepage.get("secondary_slot"), "Explore more")

recommended_product = safe_value(
    homepage.get("recommended_product"),
    "Personalized content"
)

if recommended_product in ["Personal Loan", "SME Loan", "B2B Finance"]:
    offer_description = (
        "This product has been selected based on your profile, "
        "eligibility and predicted product interest."
    )
    offer_note = "Illustrative personalized recommendation · Subject to applicable checks"

elif recommended_product == "No credit offer":
    offer_description = (
        "We are currently not displaying a credit offer. "
        "You can still explore useful financial content."
    )
    offer_note = "Your homepage has been adjusted using the configured business rules."

elif recommended_product == "Re-engagement content":
    offer_description = (
        "Explore useful updates and discover products and services "
        "that may be relevant to you."
    )
    offer_note = "Personalized engagement content"

else:
    offer_description = (
        "Explore products and financial content selected "
        "for your current customer profile."
    )
    offer_note = "Personalized homepage experience"


render_offer(
    primary,
    offer_description,
    offer_note
)


# Supporting homepage content
st.markdown(
    '<div class="section-heading">Explore more for you</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

with left:
    render_content_card(
        secondary,
        "Supporting content selected by the personalization engine.",
        "PERSONALIZED CONTENT"
    )

with right:
    render_content_card(
        "Explore financial solutions",
        "Discover products and services available in our product catalogue.",
        "EXPLORE"
    )


# Personalized banner
banner = safe_value(
    homepage.get("banner"),
    "Explore the latest updates and financial solutions."
)

st.markdown(f"""
<div class="banner-card">
    <div class="offer-eyebrow" style="color:#526D9A;">
        A MESSAGE FOR YOU
    </div>
    <div style="font-size:16px;font-weight:700;margin-top:7px;">
        {html.escape(banner)}
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. CUSTOMER PROFILE
# =========================================================

st.markdown(
    '<div class="section-heading">Your profile</div>',
    unsafe_allow_html=True
)

profile1, profile2, profile3 = st.columns(3)

with profile1:
    st.metric(
        "Customer type",
        safe_value(customer.get("customer_type"))
    )

with profile2:
    income = safe_number(customer.get("monthly_income"))
    st.metric("Monthly income", f"₹{income:,.0f}")

with profile3:
    tenure = safe_number(customer.get("tenure_months"))
    st.metric("Relationship tenure", f"{tenure:.0f} months")


# =========================================================
# 9. INTERVIEW / ADMIN DETAILS
# =========================================================

with st.expander("Decision details · Interview demonstration"):


    st.markdown("---")
    st.markdown("#### Why did the models give this customer these scores?")

    st.caption(
        "SHAP shows which features influenced each model's output. "
        "Positive contributions push the model score higher; negative "
        "contributions push it lower. Values are on the model's raw "
        "output scale, not percentage points or causal effects."
    )

    try:
        explainers = load_shap_explainers()

        product_tabs = st.tabs([
            "Personal Loan",
            "SME",
            "B2B"
        ])

        product_codes = ["pl", "sme", "b2b"]

        for tab, product_code in zip(product_tabs, product_codes):
            with tab:
                explanation = explain_customer(
                    customer=customer,
                    product_code=product_code,
                    models=models,
                    explainers=explainers,
                    top_n=5
                )

                st.dataframe(
                explanation,
                width="stretch",
                hide_index=True)

                chart_data = explanation.set_index("Feature")[
                    ["SHAP contribution"]
                ]

                st.bar_chart(chart_data)

    except Exception as error:
        st.warning(f"Could not generate the SHAP explanation: {error}")


    st.markdown("#### Customer segmentation")
    st.write(f"Assigned cohort: **{homepage.get('cohort', 'Not available')}**")

    st.markdown("#### Product propensity scores")

    score1, score2, score3 = st.columns(3)

    with score1:
        st.metric(
            "Personal Loan",
            f"{safe_number(customer.get('pl_propensity')):.2%}"
        )

    with score2:
        st.metric(
            "SME Loan",
            f"{safe_number(customer.get('sme_propensity')):.2%}"
        )

    with score3:
        st.metric(
            "B2B Finance",
            f"{safe_number(customer.get('b2b_propensity')):.2%}"
        )

    st.markdown("#### Business-rule decision")
    st.write(homepage.get("decision_reason", "Not available"))

    st.write(
        f"Selected recommendation: "
        f"**{homepage.get('recommended_product', 'Not available')}**"
    )

    st.markdown("#### Eligibility")

    eligibility_columns = {
        "Personal Loan": "eligible_pl",
        "SME Loan": "eligible_sme",
        "B2B Finance": "eligible_b2b",
    }

    eligibility_data = []

    for product_name, column_name in eligibility_columns.items():
        if column_name in customer.index:
            value = str(customer[column_name]).strip().lower()
            eligible = value in {"1", "1.0", "true", "yes"}

            eligibility_data.append({
                "Product": product_name,
                "Eligible": "Yes" if eligible else "No"
            })

    if eligibility_data:
        st.dataframe(
            pd.DataFrame(eligibility_data),
            use_container_width=True,
            hide_index=True
        )

    delinquent = str(
        customer.get("delinquent_flag", False)
    ).strip().lower() in {"1", "1.0", "true", "yes"}

    emi_blocked = str(
        customer.get("emi_card_blocked", False)
    ).strip().lower() in {"1", "1.0", "true", "yes"}

    st.write(
        f"Delinquency flag: **{'Yes' if delinquent else 'No'}**"
    )
    st.write(
        f"EMI card blocked: **{'Yes' if emi_blocked else 'No'}**"
    )

    st.caption(
        "This is a portfolio demonstration using synthetic customer data. "
        "Propensity scores are not credit approval decisions."
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Persona · Dynamic Homepage Personalization Prototype · "
    "Synthetic customer data"
)
