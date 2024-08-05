# Macroeconomic-ABM
This work presents Macroeconomic ABM using the MESA library that replicates the functioning of a real economy, aimed at analysing the effectiveness of the agent-based approach in capturing the effects of various economic policies. 

## 1. Goal
The initial goal is to create a realistic model capable of mimicking patterns observed in the real world. The model should replicate trends seen in developed economies, particularly those in European countries. Therefore, key economic indicators such as GDP, unemployment rates, inflation, and wealth distribution resulted in the simulation, should align with the actual levels exhibited by real-world economies. Additionally, the model should demonstrate a certain level of organisation in relation to the functioning of the economy, resulting from the individual behaviours and interactions within the simulation.

The second objective involves introducing economic shocks into the simulation to alter the dynamics and behaviour of individuals within the model. This approach aims to analyse the collective outcomes resulting from the introduction of a policy measure. As a result, the model is expected to react to these policy-driven shock by making the economy evolve from one state to another, thereby generating various responses in macroeconomic variables.  By doing this, the model will allow for the evaluation of the overall impact of these economic shocks.


### 2. ABM Outline
The ABM adopted in this work provides a model that is simple to use and interpret to examine macroeconomic outcomes. To achieve this, the structure of the ABM follows the bottom-up macroeconomic model developed by Delli Gatti et al (2011).

#### 2.1	Environment

2.1.1 Agents
The model represents a closed economy populated with a limited number of agents belonging to three different classes according to their macroeconomic role: 
•	Households: Agents representing the workers and consumers within the economy.
•	Firms: Agents responsible for transforming labour into goods.
•	Banks: Entities that provide liquidity to firms.

2.1.2 Structure of each agent
In this model, agents are endowed with a set of attributes, referred to as state variables. Given that individuals within a class have the same role in the economy, agents are assigned specific attributes and actions based on the class they belong to.

2.1.3 Network
The interactions among the different classes of agents are organised within three types of markets. These markets are the Labour Market, the Goods Market and the Credit Market, where a specific set of actions occur in each of them.
In the Labour Market firms calculate their labour needs based on their production levels, and households offer work in exchange for salary. The dynamics of this market involve firms posting job vacancies, adjusting wage offers in response to market conditions, and hiring workers. Alternatively, in the Goods Market households use part of their wealth to purchase goods, while firms set prices based on their production costs and market conditions. Lastly, the Credit Market is the place where firms in need of funding to perform their economic activity seek financial resources, whereas banks offer loans at different rates, in relation to the market conditions and financial capabilities of the firms applying for loans.



#### 3. ABM Implementation 
3.1 Methodology: MESA
The ABM developed in this work has been implemented using the Python programming language, specifically employing the Mesa library. This library is a Python framework for agent-based modelling that has been extensively used for analysing agent behaviour across various fields. 
Mesa enables the definition of agent-based models by creating objects that represent agents interacting with one another and with their environment (Masad & Kazil, 2015). For this work, the main components used include the agent scheduler, which handles time components and stablishes the order of the interactions within the model; and the data collector, which collects and extracts specified model- and agent-level data each time the model is run.

3.2 Parameters setting
The  initialization parameters introduced in the model are obtained from Delli Gatti’s work (2011) and others defined through calibration to ensure the model functions correctly. 

3.3	Scaling down methodology
In order to achieve a manageable size for the simulation, real data from the Spanish economy in 2023 were scaled down to determine the number of each class of agents to be included in the model. The number of workers, corresponding to the total active population, and the number of active firms, were sourced from the Instituto Nacional de Estadística (INE). The list of financial entities was obtained from the Bank of Spain. After obtaining the data, a scaling factor of 1:50,000 was applied to the figures.
