import requests
import pandas as pd
import datetime as dt

def extract_rows(item):
    """
    Function that converts a QBO API response into a dataframe
    """
    extracted_rows = []
    if 'ColData' in item:
        amount = item['ColData'][1].get('value') if len(item['ColData']) > 1 else 0
        extracted_rows.append({
            "LineItem": item['ColData'][0].get('value'),
            "Amount": float(amount) if (amount is not None and amount != '') else 0
        })
    elif isinstance(item, dict):
        for _, subitem in item.items():
            if isinstance(subitem, dict) or isinstance(subitem, list):
                exrows = extract_rows(subitem)
                extracted_rows += exrows
    elif isinstance(item, list):
        for _, subitem in enumerate(item):
            if isinstance(subitem, dict) or isinstance(subitem, list):
                exrows = extract_rows(subitem)
                extracted_rows += exrows
    return extracted_rows

def get_qbo_report(realm_id, access_token, report_name, start_date: dt.datetime, end_date: dt.datetime):
    BASE_URL = "https://quickbooks.api.intuit.com"
    assert start_date < end_date
    assert report_name in ["ProfitAndLoss", "BalanceSheet"]
    payload = {
        "accounting_method": "Accrual",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/v3/company/{realm_id}/reports/{report_name}"
    res = requests.get(url, headers=headers, params=payload)
    resobj = res.json()
    rows = extract_rows(resobj)
    df = pd.DataFrame(rows)
    return df

# This function will fetch data from QuickBooks API based on the given parameters
def fetch_qbo_data(realm_id, access_token, report_name, start_date, end_date):  
    df = get_qbo_report(realm_id, access_token, report_name, start_date, end_date)
    return df
