The value of international students to the United States. Probability of getting a non-immigrant visa.

Project timeline: 
Jan 2021 - April 2021

Project team:
- Zinaida Dvoskina (myself)
- Kirill Ilin
- Johnathan Conley
- Cindy Ye Fung

Analyzed publicly available data on the U.S. non-immigrant visa acquisition, seeking to understand if various factors such as the passport that the student holds, local location of the applicant, type of visa the student is seeking, the political party in office, wage rate, job title, etc., determine the likelihood of an international student obtaining a working visa.

To conduct research, used publicly available data from the U.S. Citizenship and Immigration Services (the number of visas issued per country, category, the political party in office, and year) and from the U.S. Department of Labor Office of Foreign Labor Certification’s case management systems (information about employment-based immigration applications: applicant’s received dates, decision dates, the most recent date a case determination decision was issued, etc.).

Tableau visualizations showed no strong trend to justify that the political party in office affects the likelihood of a foreigner obtaining a visa. A timelapse, also created in Tableau, shows the world map, where visa numbers can be filtered by region, country, and compared between years.

Created a KNN model for classification with the following variables as predictors: Received month, Agent representing employer, Annual wage rate, Annual prevailing wage, PW wage level, H-1B dependent status, Support H1B status. 

Datasets are populated with approved results of visa applications - almost 97% of data is positive, whereas the negative Case Status percentage is very small. That resulted in highly biased prediction models towards positive outcomes, which means the model wasn’t very trustworthy, even though it performed very well predicting positive outcomes for visa approval.

One of the possible ways to solve such a problem is to implement undersampling of the dataset, prior to running any prediction models. Random undersampling randomly eliminated data points from the dominating population and could even out the number of positive and negative outcomes for a more correct prediction. Due to computing power, the number of predictors was limited to 3: Full Time Position, PW, and New Employer, and the model was only run for 2020. 

A new KNN model was run on undersampled data, showed more realistic results, not biased towards a positive outcome. Chosen predictors have an impact on visa decisions, however, they only work in approximately 60% of situations. Further increase in the number of predictors may improve the model.

An interesting finding was that software engineers are at the top job title to obtain a working visa; however, they have the most denials.

__________________________________

In this repository you can find our code, Tableau workbooks, project report and a presentation with our major findings.
The data file is too big to upload here.
