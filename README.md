# IABAC-PROJECT
 INX Future Inc
 Employee Performance Analysis - INX Future Inc
Candidate Name      : Sankar S
E-Mail              : sansandeep2727@gmail.com
Project Code        : 10281 
REP Name            : DataMites™ Solutions Pvt Ltd
Assesment ID        : E10901-PR2-V18
Module              : Certified Data Scientist - Project
Exam Format         : Open Project- IABAC™ Project Submission
Registered Trainer  : Ashok Kumar A
Submission Deadline Date:   20-April-2019  

Project Summary
The Data science project which is given here is an analysis of employee performance form INX Furture Inc. The project goal is to find the performance rating of the employees from each feature of their data such as total work experience, gender, department, current role..etc. The Goal and Insights of the project as follows,

Department wise performances
Top 3 Important Factors affecting employee performance
A trained model which can predict the employee performance based on factors as inputs. This will be used to hire employees
Recommendations to improve the employee performance based on insights from analysis.
The given data of employees has the 1200 data to perform a higher level machine learning where it is well structured. The features present in the data are 28 in total. The Shape of the data is 1200x28. The 28 features are classified into quantitative and qualitative where 16 features are qualitative and 11 features are quantitative. The employee ID data is alphanumerical data which doesn't play a role as a relevant feature for performance rating.

The dataset is a complete labelled data and categorical which decides the machine learning algorithm to be used. The important aspects of the data are depending on the correlation of data between features and performance rating. The analysis of the project has gone through the stage of distribution analysis, correlation analysis and analysis by each department to satisfy the project goal.

The machine learning model which is used in this project is random forest classifier and gradient boosted classifier which predicted the nearby higher accuracy of 92%. Since it is categorical labelled data, it has to go through the classifier machine learning techniques which will be suitable for this structured data. The numerical features are the most relevant in the model according to correlation technique.

One of the goals of this project is to find the important feature affecting the performance rating. The important features were predicted using the machine learning model feature importance technique. The main technique used in the preprocessing data using the one hot encoding method to convert the string-categorical data into numerical data, because, the most of machine learning methods are based on numerical methods where strings are not supportive. The overall project was performed and achieved the goals by using the machine learning model and visualization techniques.

1. Requirement
The data was given from the IABAC for this project where the collected source is IABAC. The data is based on INX Future Inc, (referred as INX ). It is one of the leading data analytics and automation solutions provider with over 15 years of global business presence. The data is not from the real organization. The whole project was done in Jupiter notebook with python platform.

2. Analysis
Data were analyzed by describing the features present in the data. the features play the bigger part in the analysis. the features tell the relation between the dependent and independent variables. Pandas also help to describe the datasets answering following questions early in our project. The futures present in the data are divided into numerical and categorical data.

Categorical Features
These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based. The categorical features as follows,

Gender
EducationBackground
MaritalStatus
EmpDepartment
EmpJobRole
BusinessTravelFrequency
EmpEducationLevel
EmpEnvironmentSatisfaction
EmpJobInvolvement
EmpJobLevel
EmpJobSatisfaction
OverTime
EmpRelationshipSatisfaction
EmpWorkLifeBalance
Attrition
PerformanceRating
Numerical Features
These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based. The Numerical Features as follows,

Age
DistanceFromHome
EmpHourlyRate
NumCompaniesWorked
EmpLastSalaryHikePercent
TotalWorkExperienceInYears
TrainingTimesLastYear
ExperienceYearsAtThisCompany
ExperienceYearsInCurrentRole
YearsSinceLastPromotion
YearsWithCurrManager
Alphanumeric Features
Numerical, alphanumeric data within same feature. These are candidates for correcting goal. Employee ID number is a mix of numeric and alphanumeric data types.

Distribution of Numerical Features
This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.the distribution can be derived or visualized using the density map between the numerical or categorical features present in the data.

The age distribution is starting from 18 to 60 where the most of the employees are lying between 30 to 40 age count.
The distance from home to office is distributing from 0 unit to 30 unit which can be kilometre or mile. The most of the employees are coming from the range of 0 to 5 units.
Employees are worked in the multiple companies up to 8 companies where most of the employees worked up to 2 companies before getting to work here.
The hourly rate range is 65 to 95 for majority employees work in this company.
In General, Most of Employees work up to 5 years in this company.
Most of the employees get 11% to 15% of salary hike in this company.
Distribution of Categorical Features
The Gender variance is divided by 60% of Male employees and 40% of Female employees in the company.
The number of the educational backgrounds present in the employees is six unique backgrounds.
nineteen unique employee job roles are present in this company.
The most of the employees are having the education level of 3
The Job satisfaction level in this company is high level for the majority of employees.
The 85% of employees are not having attrition in their work
only 11% of employees in the company were achieved level 4 - performance rating
The overall percentage of employees doing overtime is 30%
Data Clean Check
The Data cleaning and wrangling is the part of the Data science project where the workflow the project go through this stage. because the damaged and missing data will lead to the disaster in the accuracy and quality of the model. If the data is already structured and cleaned, there is no need for the data cleaning. In this case, the given data is well structured and cleaned and there are no missing data present in this data.

Analysis by Visualization
we can able to perform the analysis by the visualisation of the data in two forms here in this project. One is by distributing the data and visualize using the density plotting. The other one is nothing but the correlation method which will visualise the correlation heat map and we can able to achieve the correlation values between the numerical features.

(i) Distribution Plot
In general, one of the first few steps in exploring the data would be to have a rough idea of how the features are distributed with one another. To do so, we shall invoke the familiar kdeplot function from the Seaborn plotting library. The distribution has been done by both numerical and categorical features. it will show the overall idea about the density and majority of data present in a different level.

(ii) Correlation Plot
The next tool in a data explorer's arsenal is that of a correlation matrix. By plotting a correlation matrix, we have a very nice overview of how the features are related to one another. For a Pandas data frame, we can conveniently use the call .corr which by default provides the Pearson Correlation values of the columns pairwise in that data frame. The correlation works bet for numerical data where we are going to use all the numerical features present in the data.

From the above Pearson correlation heat plot, we can be to see that correlation between features with numerical values in the dataset. The heat signatures show the level of correlation from 0 to 1. from this distribution we can derive the facts as follows,

The Total years of experience and job level are having the higher correlation when comparing to all features.

Experience years at this company and years with the current manager has the second higher relation between these features.

Experience years at this company and experience in the current role makes the sense of correlation.

People who have more experience with the company has the more probability to get the promotion from the correlation between them.

In this plot, the age has the important role in the total number of work experience of an employee where it is a universal truth.

Machine Learning Model
The machine learning models used in this project are

1.Random Forest classifier
2.Gradient boosted Classifier
Both machine learning algorithms are best for classification and labelled data. The train and test data are divided and fitted into the model and passed through the machine learning. Since we have already noted the severe imbalance in the values within the target variable, we implement the SMOTE method in the dealing with this skewed value via the learn Python package. The predicted data and test data achieved the accuracy rate of,

Random Forest classifier: 92%
Gradient boosted Classifier: 91%
3. Summary
The machine learning model has been fitted and predicted with the accuracy score. The goal of this project is nothing but the results from the analysis and machine learning model.

Goal 1: Department wise performances
In department wise performance, we have to analyze the data from each department present in the category. The data frame has to be separated or sliced according to department wise. In Employee department feature there are six departments available. The performance analysis by the department as follows,

Sales: The Performace rating level 3 is more in the sales department. The male performance rating the little bit higher compared to female. The total work experience does not count the performance rating.

Human Resources: The majority of the employees lying under the level 3 performance. The older people are performing low in this department. The female employees in HR department doing really well in their performance. The total work experience does matter to performance in this department.

Development: The largest number of employees are level 3 performers. Employees of all age are performing at the level of 3 only. The gender-based performance is nearly same for both.

Data Science: The highest average of level 3 performance is in data science department. Data science is the only department where less number of level 2 performers. The overall performance is higher compared to all departments. The age does not count as an important factor in their performance. Male employees are doing good in this department. Same like HR, the number of work experience does matter.

Research & Development: The age factor is not deviating from the level of performance here where different employees with different age are there in every level of performance. The R&D has the good female employees in their performance.

Finance: The finance department performance is exponentially decreasing when age increases. The male employees are doing good. The experience factor is inversely relating to the performance level.

Goal 2: The Top 3 important features affecting the employee performance
The Random Forest classifier and Gradient boosted classifier in Sklearn also contains a very convenient and most useful attribute feature importances which tells us which features within our dataset has been given most importance through the ML. The top three important features affecting the performance rating are ordered with their importance level as follows,

1. Employment Environment Satisfaction
2. Employee Salary Hike Percentage
3. Years Since the last Promotion

Goal 3: A Trained model which can predict the employee performance
The trained model is created using the machine learning algorithm as follows with the accuracy score,

Random Forest classifier: 92% accuracy
Gradient boosted Classifier: 91% accuracy
Goal 4: Recommendations to improve the employee performance
The overall employee performance can be achieved by employee environment satisfaction. The company needs to focus more on the employee environment.
The salary hike will give the boost to the employees to perform well financially and psychologically.
The promotion will help the employees to achieve more performance by giving the chance to be more responsible and leadership qualities.
The experience years in current role need to be revised while offering the employment to the new employees.
Employee's work-life balance affects the performance rating.
While recruiting for HR, consider the female candidates where they perform well compared to male.
The development and data science department is having an overall higher performance comparing to rest of the departments.
