
-- 1. CUSTOMER DISTRIBUTION BY PERSONALIZATION COHORT

SELECT
    cohort,
    COUNT(*) AS customer_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_percentage
FROM homepage_results
GROUP BY cohort
ORDER BY customer_count DESC;


-- 2. RECOMMENDATION DISTRIBUTION

SELECT
    recommended_product,
    COUNT(*) AS recommendation_count
FROM homepage_results
GROUP BY recommended_product
ORDER BY recommendation_count DESC;


-- 3. AVERAGE PROPENSITY SCORE BY COHORT

SELECT
    cohort,
    ROUND(AVG(pl_propensity), 4) AS avg_pl_propensity,
    ROUND(AVG(sme_propensity), 4) AS avg_sme_propensity,
    ROUND(AVG(b2b_propensity), 4) AS avg_b2b_propensity
FROM homepage_results
GROUP BY cohort
ORDER BY cohort;


-- 4. DELINQUENCY AND CREDIT-OFFER SUPPRESSION

SELECT
    CASE
        WHEN LOWER(CAST(delinquent_flag AS VARCHAR))
             IN ('1', '1.0', 'true', 'yes')
        THEN 'Delinquent'
        ELSE 'Not delinquent'
    END AS delinquency_status,
    COUNT(*) AS customer_count,
    SUM(
        CASE
            WHEN recommended_product = 'No credit offer'
            THEN 1
            ELSE 0
        END
    ) AS no_credit_offer_count
FROM homepage_results
GROUP BY delinquency_status
ORDER BY delinquency_status;


-- 5. EMI PROMOTIONAL SUPPRESSION

SELECT
    CASE
        WHEN LOWER(CAST(emi_promo_suppressed AS VARCHAR))
             IN ('1', '1.0', 'true', 'yes')
        THEN 'Suppressed'
        ELSE 'Not suppressed'
    END AS emi_promotion_status,
    COUNT(*) AS customer_count
FROM homepage_results
GROUP BY emi_promotion_status
ORDER BY emi_promotion_status;


-- 6. RECOMMENDATION DISTRIBUTION BY COHORT

SELECT
    cohort,
    recommended_product,
    COUNT(*) AS customer_count
FROM homepage_results
GROUP BY cohort, recommended_product
ORDER BY cohort, customer_count DESC;


-- 7. AVERAGE RECOMMENDED PROPENSITY BY PRODUCT

SELECT
    recommended_product,
    COUNT(*) AS customer_count,
    ROUND(AVG(recommended_score), 4) AS avg_recommended_score
FROM homepage_results
WHERE recommended_score IS NOT NULL
GROUP BY recommended_product
ORDER BY customer_count DESC;
