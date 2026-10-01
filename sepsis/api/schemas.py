from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Observation(BaseModel):
    hour: int = Field(ge=1, description="Hours since ICU admission (ICULOS)")
    HR: Optional[float] = None
    O2Sat: Optional[float] = None
    Temp: Optional[float] = Field(None, description="Celsius")
    SBP: Optional[float] = None
    MAP: Optional[float] = None
    DBP: Optional[float] = None
    Resp: Optional[float] = None
    Lactate: Optional[float] = None
    WBC: Optional[float] = None
    Creatinine: Optional[float] = None
    Platelets: Optional[float] = None


class PatientHistory(BaseModel):
    patient_id: str
    age: float = Field(ge=0, le=120)
    gender: int = Field(ge=0, le=1, description="1 = male, 0 = female")
    observations: List[Observation] = Field(min_length=1, max_length=336)

    @model_validator(mode="after")
    def hours_unique(self):
        hours = [o.hour for o in self.observations]
        if len(set(hours)) != len(hours):
            raise ValueError("duplicate hours in observations")
        return self


class BatchRequest(BaseModel):
    patients: List[PatientHistory] = Field(min_length=1, max_length=500)


class Driver(BaseModel):
    variable: str
    feature: str
    value: Optional[float]
    shap: float
    description: str


class Prediction(BaseModel):
    patient_id: str
    hour: int
    risk_score: float
    threshold: float
    alert: bool
    risk_level: str
    alert_message: str
    top_drivers: List[Driver]
    model_version: str
