# Production Status Detection - Analysis Summary

## Current Situation

Based on analyzing your three CSV files (`metadata.csv`, `processdata.csv`, `productiondata.csv`), here's what I found:

### Data Available

| Table | Key Columns | Use Case |
|-------|------------|----------|
| **metadata.csv** | Marking, Recipe, Description, Datetime | NEW items created during production runs |
| **processdata.csv** | Sequence_nr, Marking, Step_nr, Set_time, **Actual_time** | Step execution times (0 = not completed yet) |
| **productiondata.csv** | Marking, Step_nr, Type, Value, Datetime | Sensor/process data with Type (0, 1, 2) |

---

## Key Findings

### 1. **NO Direct "IDLE/STOP/RUN" Status Column**
❌ The data does NOT contain a direct column like "ProductionStatus" or "IsIdle"

However, we can **infer** the status from behavior patterns:

### 2. **Three Ways to Detect Production Status**

#### Option A: New Items in METADATA (⭐ BEST for your use case)
```
- Production ACTIVE: New rows appear in metadata regularly
- Production IDLE: No new rows for extended period
```
**Current data:**
- Last metadata entry: **May 7, 2026 @ 07:21:15**
- Most recent items: Items B13645, B13644, B13633

**Advantage:** Simple, reliable, tells if production line is running

---

#### Option B: Latest PRODUCTIONDATA Timestamp
```
- Production ACTIVE: Regular sensor data updates
- Production IDLE: Last update > 30 mins ago
```
**Current data:**
- Latest productiondata: **May 7, 2026 @ 08:27:31** (sensors still running)

**Advantage:** Most up-to-date indicator

**Caveat:** Could indicate system problem if sensors keep running but no new items

---

#### Option C: PROCESSDATA Actual_time
```
- Check if new Steps are being logged with Actual_time > 0
- Actual_time = 0 means step hasn't completed yet
```
**Current data:**
- Last 10 steps all have Actual_time > 0 (completed)
- Only 1,921 / 292,658 steps have Actual_time = 0 (incomplete)

**Advantage:** Shows if active production is happening

---

## Recommendation for Your 30-Minute Alert System

### The Challenge
You need to distinguish between:
1. ✅ **Intentional IDLE**: Production line is off, no alert needed
2. ❌ **SYSTEM PROBLEM**: Data not being sent, alert needed

### Proposed Solution

```python
def check_production_status(metadata_df, productiondata_df):
    """
    Determine if production is idle (intentional) or if there's a system problem
    """
    
    # Get the latest timestamps
    latest_metadata_time = pd.to_datetime(metadata_df['Datetime']).max()
    latest_productiondata_time = pd.to_datetime(productiondata_df['Datetime']).max()
    
    current_time = datetime.now()
    metadata_gap_minutes = (current_time - latest_metadata_time).total_seconds() / 60
    productiondata_gap_minutes = (current_time - latest_productiondata_time).total_seconds() / 60
    
    # Decision logic
    if metadata_gap_minutes > 30:
        # No new items in 30 minutes = production likely IDLE
        return {
            'status': 'IDLE',
            'alert': False,  # Don't send alert - production is just not running
            'reason': f'No new metadata entries for {metadata_gap_minutes:.0f} minutes'
        }
    
    elif metadata_gap_minutes <= 30 and productiondata_gap_minutes > 30:
        # New items exist but sensor data is delayed = SYSTEM PROBLEM!
        return {
            'status': 'ERROR',
            'alert': True,  # Send alert - there's a problem!
            'reason': f'Metadata updated {metadata_gap_minutes:.0f} min ago, but productiondata is {productiondata_gap_minutes:.0f} min old'
        }
    
    else:
        # All systems nominal
        return {
            'status': 'RUNNING',
            'alert': False,
            'reason': 'All data streams current'
        }
```

---

## Summary Table

| Situation | Metadata Gap | ProductionData Gap | Action |
|-----------|-------------|-----------------|--------|
| Production Running Normally | < 5 min | < 5 min | ✅ No alert |
| Production Idle (Intentional) | > 30 min | > 30 min | ✅ No alert |
| **SYSTEM PROBLEM** | < 30 min | > 30 min | 🚨 **SEND ALERT** |
| System Issues | > 30 min | > 30 min | ⚠️ Maybe alert if unusual |

---

## Next Steps

Would you like me to:
1. ✅ Create a full alert system script with email integration?
2. ✅ Analyze your database schema to see if there's an actual "Status" column in the original database?
3. ✅ Check if the Excel file `QC progress overview` has any status information?
4. ✅ Set up a scheduled job that checks this periodically?

Let me know how you'd like to proceed!

