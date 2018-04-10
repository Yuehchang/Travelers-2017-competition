## Travelers-2017-competition

### About Travelers competition:
1. Business problems: 
  * Our clients were expected to identify the policies that will cancel before the end of their term.
  * Our missions were to understand the correlation between cancellation and features of policyholders.
2. Data sets: 
  * The data set is based on 4 years of property insurance policies from 2013 to 2016. 
  * There are roughly 7500 policies in the training data and each policy only has one observation. 
  * There are almost 2000 policies that were canceled during the effective term. 

### About the process of building the models:
1. Variable describtion:
  * Categorical variables: Gender, Salse channel, Coverage type, Dwelling type, Credit, House color, Zipcode
  * Numeric variables(Continuos): Premium, Length at residence, Age
  * Numeric variables(Discrete): Tenure, Claim, # adutls, # children, Marital stutus, Year

2. Data preparation:
  * Correlating (Understanding which features contribute significantly to the concellation)
  
  * Correcting (Dropping not contributed features, deleting outlier in sepcific features)
    1. Indentified unreasonable age such as 100 in dataset; therefore, deleted rows above 75 years old(>0.99 percentile).
    2. Indentified -1 in our target feature(cancellation=0 or 1); therefore, removed rows with -1 in cancellation. 
    3. Kept n-1 dummy to prevent dummy variable trap in training model.     
    
  * Completing (Inputting missing value)
    1. Inputted globel mean for numeric variables
    2. Inputted "Other" for categorical variables
  
  * Creating (Create new features from exsiting variable, such as dummy, zipcode to state name)
    1. Created zip code list of State level using [uszipcode package](https://pypi.python.org/pypi/uszipcode/0.1.3), and mapped zip code into State abbreviations. 
    2. Created dummy for all categorical features. 
    3. Converted categorical variable(credit) to ordinal variable. 
 
 3. Model building:
   * Applied scikit-learn package for building models, including:
     * LogisticRegression
     * Support Vector Machine
     * Gradient Boosting 
 
 4. Final resutls:
    The final models we built got the **highest accuracy 0.72** among all the competitors (1st / 20)

