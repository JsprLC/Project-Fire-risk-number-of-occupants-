# Visualizing Small Risk Values - Quick Reference

## Problem
Your predicted fatalities values are very small per building (e.g., 2.47e-04), which makes visualization challenging with standard approaches.

## Solution: Percentile-Based coloring

**Best for:** When absolute values are too small to discriminate visually

**Key Features:**
- ✅ **Percentile-based coloring** - Buildings ranked 0-100%
- ✅ **6 Risk Categories:**
  - Very High (Top 5%) - Dark Red
  - High (90-95%) - Crimson
  - Elevated (75-90%) - Orange
  - Moderate (50-75%) - Gold
  - Low-Moderate (25-50%) - Yellow-Green
  - Low (Bottom 25%) - Green
- ✅ Enhanced popups showing both absolute values AND percentile rank
- ✅ Category-wise statistics (buildings, deaths, occupants per category)
- ✅ Better visual discrimination for small values

---

## Understanding Small Values

For values like `expected_deaths_mean = 2.47e-04`:

**What it means:**
- 2.47e-04 = 0.000247 = 0.0247%
- Approximately 2.47 deaths per 10,000 exposures
- Very low absolute risk, but relative comparisons still matter

**Why percentile-based visualization helps:**
- Even if all values are small, some buildings are still riskier than others
- Percentiles show: "This building is in the top 10% of risk"
- Easier to identify priorities for intervention

---

## Interpretation Guide

### For Small Absolute Values

**Focus on relative risk:**
- Don't worry about absolute magnitude
- Compare buildings to each other
- Use percentiles for prioritization

**Key questions to answer:**
1. Which buildings are in the top 10% of risk?
2. How does uncertainty vary across buildings?
3. Are high-risk buildings also high-occupancy?
4. What building characteristics correlate with risk?

### Coefficient of Variation (CV)

CV = Std / Mean measures relative uncertainty:
- **CV < 0.3**: Reliable estimates
- **CV 0.3-0.7**: Moderate uncertainty
- **CV > 0.7**: High uncertainty, need more data

For example: CV = 5.05e-05 / 2.47e-04 = 0.204 (reliable)

---

## Tips for Analysis

### Prioritization Strategy

1. **Immediate attention (Very High Risk):**
   - Top 5% by expected deaths
   - High occupancy (>50 people)
   - Low uncertainty (CV < 0.5)

2. **Further investigation (High/Elevated):**
   - 75th-95th percentile risk
   - High uncertainty (CV > 0.5)
   - Medium-high occupancy

3. **Monitor (Moderate and below):**
   - Below 75th percentile
   - Lower occupancy
   - Any building characteristics

---

## Example Workflow

```bash
# 1. Create percentile-based interactive map
python Final_visualize_risk.py

# 2. Open the HTML file and explore
firefox building_risk_percentile_map.html

# 3. Generate publication figures
python visualize_risk_static.py

# 4. Review statistics and identify priorities
# Check console output for category distributions
```

---

**Remember:** Even with small absolute values, relative risk matters for prioritization!
