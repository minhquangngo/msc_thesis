## WRDS database set-up
**Great resources to get started/ refer**:
- (Retrieving historical members S&P500)[https://library.smu.edu.sg/topics-insights/notes-and-thoughts-retrieving-historical-members-sp-500-wrds]
- 
**Steps**
- Create a .pgpass file to store username and password
- pip install wrds
**Notes:**
- As of July 202, Compustat no longer has S&P Dow Jones Indices. Access it through CRSP.
- For data after February 2025, you will need to change the data tables to their "*_V2" counterpart.
