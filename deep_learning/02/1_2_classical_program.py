def check_loan_eligibility(age, income):
    if age < 18 or age > 65:
        return 'NO'
    if income < 30000:
        return 'NO'
    return 'YES'

if __name__ == '__main__':
    age = int(input('Please enter age: '))
    income = float(input('Please enter income: '))
    eligibility = check_loan_eligibility(age, income)
    print(eligibility)
