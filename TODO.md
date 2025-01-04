# TODO LIST

## Implementation:
- [...] Imlement Sokolov-Lyon index,
- [...] Implement MacPhie index,
- [...] Implement Cornell's index,
- [...] Implement Levis index
- [+] Keep max 3 labels per recording
- [] Make weights based on number of beats, not recordings.
- [] Make domain knowledge being compute for each beat, or at least to correlate to each beat instead of being always the average

Indexes are implemented but not yet included in the domain knowledge

## Experiments:
- [+] LSTM,(7;2),  12 LEADS, No domain knowledge, BW Removed 
- [+] LSTM,(7;2),  12 LEADS, No domain knowledge, BW NOT Removed 
- [+] LSTM,(7;2),  2 LEADS, No domain knowledge, BW Removed 
- [+] LSTM,(7;2),  2 LEADS, No domain knowledge, BW NOT Removed 
- [+] LSTM,(15;10) ,  12 LEADS, WITH domain knowledge, BW Removed 
- [+] LSTM,(15;10) ,  12 LEADS, WITH domain knowledge, BW NOT Removed 
- [+] LSTM,(15;10),  12 LEADS, No domain knowledge, BW Removed 
- [+] LSTM,(15;10),  12 LEADS, No domain knowledge, BW NOT Removed 
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at BETA branch, BW Removed
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at BETA branch, BW NOT Removed
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at BETA without PCA, BW Removed 
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at BETA without PCA, BW NOT Removed 
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at MLP, BW Removed 
- [+] LSTM,(7;2),  2 LEADS, Domain knowledge at MLP, BW NOT Removed 
- [+] Każdy feature na osobny branch - 4 branches - LSTM i NBEATS
 

## Miscallenous:
- [] copy results for NBEATS with domain knowledge, BW (NOT) Removed all leads

