import numpy as np
from scipy.stats import norm
import pandas as pd

def get_elbows(values, n=3):

    """
    Zhu & Ghodsi (2006) profile likelihood elbow detection.
    Returns up to n elbow locations.
    """

    d = np.sort(np.asarray(values))[::-1]

    elbows = []
    start = 0

    for _ in range(n):

        current = d[start:]
        p = len(current)

        if p < 3:
            break

        loglik = []

        for q in range(1, p):

            group1 = current[:q]
            group2 = current[q:]

            mu1 = np.mean(group1)
            mu2 = np.mean(group2)

            sigma2 = (
                np.sum((group1 - mu1) ** 2)
                + np.sum((group2 - mu2) ** 2)
            ) / (p - 2)

            sigma2 = max(sigma2, 1e-10)

            ll = (
                np.sum(norm.logpdf(group1, mu1, np.sqrt(sigma2)))
                + np.sum(norm.logpdf(group2, mu2, np.sqrt(sigma2)))
            )

            loglik.append(ll)

        elbow = np.argmax(loglik) + 1

        elbows.append(start + elbow)

        start += elbow

        if start >= len(d) - 2:
            break

    return elbows

def get_region(aal):
    """Parses AAL from ROI_labels.csv and returns the brain region its from"""

    if pd.isna(aal):
        return None

    aal = str(aal)

    if "Frontal" in aal:
        return "Frontal"

    elif ("Precentral" in aal or
          "Supp_Motor" in aal or
          "Rolandic" in aal):
        return "Frontal"

    elif ("Parietal" in aal or
          "Postcentral" in aal or
          "Precuneus" in aal or
          "Angular" in aal or
          "SupraMarginal" in aal):
        return "Parietal"

    elif ("Temporal" in aal or
          "Heschl" in aal or
          "Fusiform" in aal):
        return "Temporal"

    elif ("Occipital" in aal or
          "Calcarine" in aal or
          "Lingual" in aal or
          "Cuneus" in aal):
        return "Occipital"

    elif ("Cerebelum" in aal or
          "Vermis" in aal):
        return "Cerebellum"

    else:
        return "Other"