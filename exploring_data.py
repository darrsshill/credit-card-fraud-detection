

import pandas as pd          
import matplotlib.pyplot as plt  

df = pd.read_csv("../creditcard.csv")

print("File loaded successfully!")
print("The data is now stored in a variable called 'df'")


number_of_rows    = df.shape[0]
number_of_columns = df.shape[1]

print("")   
print("HOW BIG IS THE DATASET?")
print("Rows    :", number_of_rows)     # each row = one transaction
print("Columns :", number_of_columns)  # each column = one piece of info



print("")
print("COLUMN NAMES:")
print(df.columns.tolist())


print("")
print("FIRST 5 ROWS OF DATA:")
print(df.head())


counts = df['Class'].value_counts()

normal_count = counts[0]
fraud_count  = counts[1]

print("")
print("FRAUD vs NORMAL:")
print("Normal transactions :", normal_count)
print("Fraud  transactions :", fraud_count)

# Let's calculate the fraud percentage
fraud_percentage = (fraud_count / number_of_rows) * 100
print("Fraud percentage    :", round(fraud_percentage, 3), "%")

print("")
print("NOTE: The data is very unbalanced!")
print("Almost all transactions are normal. Very few are fraud.")
print("This makes it tricky for the model — we will fix this in Step 2.")




smallest_amount = df['Amount'].min()
largest_amount  = df['Amount'].max()
average_amount  = df['Amount'].mean()

print("")
print("TRANSACTION AMOUNTS:")
print("Smallest transaction : €", round(smallest_amount, 2))
print("Largest  transaction : €", round(largest_amount, 2))
print("Average  transaction : €", round(average_amount, 2))

# Now let's compare: how much was the average FRAUD vs NORMAL?
# df[df['Class'] == 1] means: "give me only the rows where Class = 1 (fraud)"
fraud_rows  = df[df['Class'] == 1]
normal_rows = df[df['Class'] == 0]

average_fraud_amount  = fraud_rows['Amount'].mean()
average_normal_amount = normal_rows['Amount'].mean()

print("")
print("Average fraud  transaction : €", round(average_fraud_amount, 2))
print("Average normal transaction : €", round(average_normal_amount, 2))

plt.figure(figsize=(6, 4))   
plt.bar(
    ['Normal', 'Fraud'],          # x-axis labels
    [normal_count, fraud_count],  # bar heights
    color=['green', 'red']        # green for normal, red for fraud
)

plt.title('Normal vs Fraud Transactions')
plt.xlabel('Transaction Type')
plt.ylabel('Number of Transactions')

# plt.tight_layout() = makes sure nothing is cut off
plt.tight_layout()

# Save the chart as an image file in the same folder
plt.savefig("step1_chart.png")

# Show the chart on screen
plt.show()

print("")
print("Chart saved as step1_chart.png")

