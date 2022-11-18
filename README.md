
### 


## Initial Focus 

The purpose of this project is to create a classification model to predict the booking cancellations worldwide for The Arrakis Hotel Group (A large chain of Internationl Hotels) and to provide new insights into the factors that influence cancellation behaviour with respect to hotel bookings. 
<hr>


## Background Information

Around the world, hotel cancellations are one of the greatest challenges facing the industry. The hotel cannot do much except sell the room at a relatively low price when there is a last-minute cancellation. As a result of last-minute check-ins made through Online Travel Agencies, there is even a higher cost involved. Consequently, profits are adversely affected. Sometimes, the hotel is not even able to sell the room at the last minute. Consequently, you lose out on the opportunity to earn higher revenue per available room. Despite cancellation fees associated with last-minute cancellations, the overall revenue that could have been generated takes a financial hit. The cancellation of hotel reservations that cannot be replaced with another guest not only results in financial losses for the hotel but also causes other issues. The number of guests determines the operational setup of a hotel, so an inaccurate forecast can lead to overstaffing and understaffing, insufficient supplies etc. 

There is no doubt that cancellations by guests are a major problem for hotels. The issue has been discussed for the past few years, but the problem still remains very specific to the hotel chain and the types of hotels.
<hr>



## Problem Statement

There has been a decline in revenue at Arrakis over the past few months due to the cancellation of reservations. Arrakis is suffering significant financial losses as a result of these cancellations.


We have briefly discussed how cancellations affect a hotel chain. Now, Arrakis would like to take additional steps to alleviate the pain of cancelled bookings. The first step is to build a model for predicting booking cancellations.
<hr>

## Stakeholders 

- Sales Team (High Power, High Interest)
- Marketing Team (Low Power, High Interest)

This model is being developed for the **Sales Team**, which is responsible for all bookings and is also responsible for managing customer needs. The **Marketing Team** will be involved when there is a high probability of cancellations and will work closely with sales to reduce the number of cancellations, they can offer promotions and other benefits to ensure that bookings are retained. 
<hr>

## Proposal

The reservation cancellations depends on a lot of factors like arrival months, reservation type, hotel type etc. Here are the list of features that can be used to predict whether a booking is likely to be cancelled or not 

> - is_canceled: Target Variable


| Feature          	| Descriptions                              	|
| :---              | :---                                          |
| hotel:         	| Hotel (H1 = Resort Hotel or H2 = City Hotel).                	|
| is_canceled:      	    | Value indicating if the booking was canceled (1) or not (0).                          	|
| lead_time: 	        | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date.                     	|
| arrival_date_year: | Year of arrival date.                 	|
| arrival_date_month: 	    | Month of arrival date.   |
| arrival_date_week_number:  | Week number of year for arrival date.         	|
| arrival_date_day_of_month:| Day of arrival date.                  	|
| stays_in_weekend_nights:    | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel.
| stays_in_week_nights:  | Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel.	|
| adults:  | Number of adults.                	|
| children:     | Number of children. 
| babies:   	| Number of babies.                  	|
| meal:      | Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner).                           	|
| country:        	| Country of Origin.            	|
| market_segment:       | Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”. 
| distribution_channel:         	| Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”.     	|
| is_repeated_guest:  | Value indicating if the booking name was from a repeated guest (1) or not (0).               	|
| previous_cancellations:  | Number of previous bookings that were cancelled by the customer prior to the current booking. |
| reserved_room_type:  | Code of room type reserved. Code is presented instead of designation for anonymity reasons.|
| assigned_room_type:  | Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons.               	|
| booking_changes:  | Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation.               	|
| deposit_type:  | Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.               	|
| agent:  | ID of the travel agency that made the booking.                	|
| company:  | ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons.               	|
| days_in_waiting_list:  | Number of days the booking was in the waiting list before it was confirmed to the customer. 	|
| customer_type:  | Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking. |
| required_car_parking_spaces: | Number of car parking spaces required by the customer. |
| total_of_special_requests: | Number of special requests made by the customer (e.g. twin bed or high floor). |





<hr>

## Performance Metrics

![Screenshot%202022-11-16%20133944.png](attachment:Screenshot%202022-11-16%20133944.png)

In this problem we are looking to minimize the **False Negatives**. We don't want to make false predictions that the reservation won't be cancelled when in fact it will. This in turn will lead to a loss of revenue per available room, which contributes to the highest profit margins in the hotel industry. In the case of last minute cancellations, Arrakis would have to list the room at a cheaper price, or there is the possibility that the room remains unsold, which would severely affect profitability. 
 
<hr> 
 
**Recall** - It is the ability of a classifier to predict the positives out of the actual positives in the data. For our classification model, it will be the percentage of predicted canceled transactions out of all the actual canceled transactions. A high recall score indicates that the model is good at identifying positive examples, which is what we are aiming for. 
 
<hr>

The **False Negative Rate** – also called the miss rate – is the probability that a true positive will be missed by the test. It’s calculated as FN/FN+TP, where FN is the number of false negatives and TP is the number of true positives (FN+TP being the total number of positives).

<hr>

The **Accuracy score** is calculated by dividing the number of correct predictions by the total prediction number. We will be using multiple Machine learning models to make our prediction, and we want to accurately make the predictions on our model.


## Specification

- Python: 3.9.12
- Pandas: 1.4.3
- Seaborn: 0.11.2
- sklearn: 1.1.2.
- Numpy: 1.21.5

This data contains 110000s  of bookings (each record represents an actual booking) and their attributes.  The target variable of interest is is_cancelled (Value indicating if the booking was canceled (1) or not (0)).  The data was provided as a part of our CIS: 508 Assignment.
<hr>


## Table of Contents: <a class="anchor" id="steps"></a>
- [1. Libraries & Custom Functions](#libraries)
- [2. Data Wrangling](#wrangle)
   - [2.1 Data Gathering](#gather)
   - [2.2 Data Assessment](#assess)
   - [2.3 Data Cleaning](#clean)

- [3. Exploratory Data Analysis](#eda)
   - [3.1 Univariate Analysis](#univariate)
   - [3.2 Outlier Analysis & Treatment](#outlier)
   - [3.3 Bivariate Analysis](#bivariate)
- [4. Feature Selection](#feature)
- [5. Model Building](#model)
    - [5.1 Model 1](#model001)
    - [5.2 Model 2](#model002)
    - [5.3 Model 3](#model003)
    - [5.4 Model 4](#model004)
    - [5.5 Model 5](#model005)
- [6. Conclusion](#conclusion)
