# banking_tools.py
import pandas as pd
import numpy as np
import datetime


def calculate_loan_schedule(principal, monthly_rate, term_months, include_tax=True):
    """
    Eşit Taksitli Kredi Hesaplama (Annuity).
    include_tax=True ise %15 KKDF + %15 BSMV (toplam %30) faize eklenir.
    (Konut kredisinde vergiler 0'dır, İhtiyaçta vardır.)
    """
    # Oran kontrolü (yüzde girildiyse ondalığa çevir)
    if monthly_rate > 1:
        monthly_rate = monthly_rate / 100

    # Vergiler (KKDF %15 + BSMV %15 = %30 artış)
    tax_multiplier = 1.30 if include_tax else 1.00
    gross_rate = monthly_rate * tax_multiplier

    if gross_rate <= 0:
        return None, "Faiz oranı 0 olamaz."

    # Taksit Formülü: P * r * (1+r)^n / ((1+r)^n - 1)
    monthly_payment = principal * (gross_rate * (1 + gross_rate) ** term_months) / ((1 + gross_rate) ** term_months - 1)

    schedule = []
    remaining_balance = principal
    total_interest_paid = 0
    total_tax_paid = 0

    start_date = datetime.date.today()

    for i in range(1, term_months + 1):
        # İçerideki faiz (Vergisiz ham faiz üzerinden hesaplanır, vergi sonradan eklenir)
        interest_part = remaining_balance * monthly_rate
        tax_part = interest_part * (tax_multiplier - 1)  # Sadece vergi kısmı

        # Toplam faiz+vergi ödemesi
        total_cost_part = interest_part + tax_part

        principal_part = monthly_payment - total_cost_part
        remaining_balance -= principal_part

        if remaining_balance < 0: remaining_balance = 0

        schedule.append({
            "Taksit No": i,
            "Tarih": (start_date + datetime.timedelta(days=30 * i)).strftime("%Y-%m-%d"),
            "Taksit Tutarı": round(monthly_payment, 2),
            "Anapara Ödemesi": round(principal_part, 2),
            "Faiz Ödemesi": round(interest_part, 2),
            "Vergi (KKDF+BSMV)": round(tax_part, 2),
            "Kalan Anapara": round(remaining_balance, 2)
        })

        total_interest_paid += interest_part
        total_tax_paid += tax_part

    df_schedule = pd.DataFrame(schedule)

    summary = {
        "Kredi Tutarı": principal,
        "Aylık Taksit": round(monthly_payment, 2),
        "Toplam Geri Ödeme": round(monthly_payment * term_months, 2),
        "Toplam Faiz": round(total_interest_paid, 2),
        "Toplam Vergi": round(total_tax_paid, 2),
        "Maliyet Oranı (Yıllık Efektif)": round(((monthly_payment * term_months - principal) / principal) * 100, 2)
    }

    return df_schedule, summary


def calculate_deposit_return(amount, days, annual_rate, withholding_tax_rate=0.075):
    """
    Mevduat Getirisi Hesaplama.
    Formül: (Anapara * Faiz * Gün) / 36500
    Stopaj (withholding_tax_rate) düşülür.
    """
    gross_return = (amount * annual_rate * days) / 36500
    net_return = gross_return * (1 - withholding_tax_rate)

    return {
        "Vade (Gün)": days,
        "Brüt Getiri": round(gross_return, 2),
        "Net Ele Geçen (Stopaj Düşülmüş)": round(net_return, 2),
        "Stopaj Tutarı": round(gross_return - net_return, 2),
        "Vade Sonu Toplam Bakiye": round(amount + net_return, 2)
    }


def dti_check(income, total_monthly_debt):
    """
    Borç / Gelir Oranı (DTI) hesaplar.
    %50 kritik eşiktir.
    """
    if income <= 0: return 0, "Gelir girilmedi"

    ratio = (total_monthly_debt / income) * 100
    status = "Uygun"
    if ratio > 50:
        status = "Yüksek Risk (Red İhtimali)"
    elif ratio > 40:
        status = "Dikkat (Ek Teminat Gerekebilir)"

    return round(ratio, 2), status