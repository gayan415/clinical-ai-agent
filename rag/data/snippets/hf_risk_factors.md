# Heart Failure Risk Factors and Prognostic Indicators

## Demographic Risk Factors
- **Age:** Risk increases significantly after age 65. Each decade of life approximately doubles heart failure incidence
- **Sex:** Males have higher incidence of HFrEF. Females more commonly develop HFpEF and typically present at older ages
- **Race:** African Americans have the highest incidence and mortality from heart failure among all racial groups in the US

## Key Prognostic Indicators (Used in Risk Models)

### Ejection Fraction (EF)
- **Normal:** 55-70%
- **Mildly reduced (HFmrEF):** 41-49%
- **Reduced (HFrEF):** ≤ 40% — qualifies for full GDMT
- **Severely reduced:** ≤ 20% — high risk, may need advanced therapies
- Each 5% decrease in EF below 40% is associated with increased mortality risk

### Serum Creatinine
- **Normal range:** 0.7-1.3 mg/dL
- **Elevated (> 1.5 mg/dL):** Indicates renal impairment, common comorbidity in HF
- **Severely elevated (> 2.0 mg/dL):** Strong predictor of poor prognosis
- Cardiorenal syndrome: heart failure causes kidney dysfunction and vice versa
- Serum creatinine > 1.5 mg/dL is an independent predictor of mortality in heart failure

### Serum Sodium
- **Normal range:** 136-145 mEq/L
- **Hyponatremia (< 135 mEq/L):** Marker of neurohormonal activation and poor prognosis
- **Severe hyponatremia (< 130 mEq/L):** Associated with significantly increased in-hospital mortality
- Persistent hyponatremia despite treatment is one of the strongest predictors of poor outcome

### Anemia
- **Defined as:** Hemoglobin < 13 g/dL (men) or < 12 g/dL (women)
- Present in 30-50% of heart failure patients
- Independently associated with increased mortality, hospitalization, and reduced functional capacity
- Causes: chronic disease, iron deficiency, renal impairment, hemodilution

### Natriuretic Peptides
- **BNP:** B-type natriuretic peptide. Normal < 100 pg/mL. Heart failure likely if > 400 pg/mL
- **NT-proBNP:** N-terminal pro-BNP. Normal < 300 pg/mL. Heart failure likely if > 900 pg/mL (age-adjusted)
- Used for diagnosis, prognosis, and treatment monitoring
- Rising trend indicates worsening heart failure even if absolute value is below threshold

## Comorbidities That Worsen Prognosis
- **Diabetes mellitus:** 2-4x increased risk of heart failure. Present in 40% of HF patients
- **Hypertension:** Leading cause of HFpEF. Long-term pressure overload causes ventricular remodeling
- **Chronic kidney disease:** Present in 50% of HF patients. Limits use of ACEi/ARB and MRA
- **Atrial fibrillation:** Present in 30-40% of HF patients. Reduces cardiac output, increases stroke risk
- **Coronary artery disease:** Leading cause of HFrEF. Ischemic cardiomyopathy from prior MI

## Follow-Up Time as a Prognostic Variable
- In clinical datasets, shorter follow-up time is often correlated with higher mortality
- Patients who die early have shorter follow-up periods by definition
- The UCI Heart Failure dataset includes a "time" variable (follow-up period in days) that is one of the strongest predictors of death event — this reflects both disease severity and survival bias
- When building predictive models, the time variable should be interpreted carefully as it encodes survival information
