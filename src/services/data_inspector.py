# src/services/data_inspector.py
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()

class DataInspector:
    """
    Enterprise-grade data inspection to prevent LLM hallucinations.
    Extracts statistical candidates (min, max, sum, cutoff values) to guide the LLM.
    """
    
    @staticmethod
    def inspect_dataframe(df: pd.DataFrame) -> dict:
        """
        Deep inspection of dataframe to find likely answers (cutoffs, sums, outliers).
        """
        if df is None or df.empty:
            return {"error": "Empty dataframe"}
            
        summary = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(5).to_dict(orient='list')
        }
        
        # Statistical candidates for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        candidates = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0: continue
            
            stats = {
                "min": float(series.min()),
                "max": float(series.max()),
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "count": int(len(series)),
                "std": float(series.std())
            }
            candidates[col] = stats
            
        summary["candidates"] = candidates
        return summary
