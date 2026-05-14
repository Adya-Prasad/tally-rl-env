## FIELDS DESCRIPTIONS:

### (1) Company Setup
1. Name
2. Mailing Name Address
3. State
4. Country
6. Pincode
7. Telephone
8. Mobile
9. Fax
10. E-mail
11. Website
12. Financial year beginning from
13. 
14. Currency
15.


### (2) Creating Ledger
1. Select company
2. From "Gateway of Tally" click [Create]
3. It will open a "List of Masters" and from "Accounting Master" click on [Ledger]
4. It will open a ui with fields:
    1. Name
    2. (alias)
    3. Under: [List of Groups]
    4. Maintain balance bill-by-bill: bool
    5. Mailing Details:
        1. Name
        2. Address
        3. State
        4. Country
        5. Pincode
    6. Banking Details:
        1. Provider bank details: Bool
            1. A/C Holder's name
            2. A/c No.
            3. IFS Code
            4. SWIFT Code
            5. Bank Name
            6. Branch
    7. Tax Registeration Details:
        1. PAN/IT No.
        2. Registeration Type
        3. GSTIN/UIN
        4. Additional GST details
    8. Opening Balance

### (3) Entering Vouchers (Capital)
- A company should be selected
- From 'Gateway of Tally' sidebar choose [Vouchers]
- Required: Correct Voucher Type [Keyboard shortcuts: Function keys]
- Required: Correct related ledger type
- Input fields: Account
- Required: Choose a Ledger

### (3) Entering Vouchers (Purchase)
- Required: A party account name
- Required: Created Purchase Ledger
- Required: Choose a Purchase Ledger
- Required: A entry in voucher
- Input fields: 
    1. Supplier Invoice No:
    2. Party A/c name
    3. Purchase Ledger
    4. 

## VOUCHER DATA MODEL
```
A voucher = {
  voucher_type: enum,
  date: date,
  voucher_number: auto-generated,
  narration: optional string,
  entries: list of (ledger_name, debit_amount, credit_amount)
}
```
`Invariant: sum(debit_amount for e in entries) == sum(credit_amount for e in entries)`

For each voucher type, the "natural" ledger sides are:
- Sales `(F8)`: Customer Dr, Sales Cr, GST Output Cr
- Purchase `(F9)`: Purchase Dr, GST Input Dr, Supplier Cr
- Receipt `(F6)`: Cash/Bank Dr, Customer Cr
- Payment `(F5)`: Vendor/Expense Dr, Cash/Bank Cr
- Contra `(F4)`: Both ledgers are Cash/Bank type
- Journal `(F7)`: freeform Dr/Cr against any ledgers