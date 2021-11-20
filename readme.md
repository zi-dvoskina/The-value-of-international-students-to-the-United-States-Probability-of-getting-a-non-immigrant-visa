The value of international students to the United States. Probability of getting a non-immigrant visa.

Project timeline: 
Jan 2021 - April 2021

Project team:
- Zinaida Dvoskina (myself)
- Kirill Ilin
- Johnathan Conley
- Cindy Ye Fung

Analyzed publicly available data on the U.S. non-immigrant visa acquisition. To conduct research, used publicly available data from the USCIS (the number of visas issued per country, category, the political party in office, and year) and from the US Department of Labor Office of Foreign Labor Certification (employment-based immigration applications: applicant’s received dates, decision dates, the most recent date a case determination decision was issued, etc.).

Created a Tableau timelapse, showing the world map, where visa numbers can be filtered by region, country, and compared between years. Other visualizations showed no strong trend to justify that the political party in office affects the likelihood of a foreigner obtaining a visa. 

![Visa Time Lapse](https://user-images.githubusercontent.com/67168908/142707523-ef4923f0-d868-4c8d-b905-23057423789f.png)
![Visa by Year and Party](https://user-images.githubusercontent.com/67168908/142707443-ceb1f772-a9ad-4f45-b090-52f464e51ee7.png)
![Visa Cat Working](https://user-images.githubusercontent.com/67168908/142707450-188090bf-b053-4bea-848e-b66c6085f485.png)

Created a KNN model for classification with the following variables as predictors: Received month, Agent representing employer, Annual wage rate, Annual prevailing wage, PW wage level, H-1B dependent status, Support H1B status. Datasets are populated with approved results of visa applications - almost 97%. That resulted in highly biased prediction models towards positive outcomes, which means the model wasn’t very trustworthy, even though it performed very well predicting positive outcomes for visa approval.

To solve the problem, randomly eliminated data points and aligned the number of positive and negative outcomes for a more correct prediction. Due to computing power, had to limit the number of predictors to 3: Full Time Position, PW, and New Employer, and the model was only run for 2020. 

A new KNN model run on undersampled data showed results not biased towards a positive outcome. Chosen predictors had an impact on visa decisions, however, only in approximately 60% of cases. Further increase in the number of predictors could improve the model.

An interesting finding was that software engineers are at the top job title to obtain a working visa; however, they have the most denials.

__________________________________

In this repository you can find our code, Tableau workbooks, project report and a presentation with our major findings.
The data file is too big to upload here.
