# Review_based_game_recommendation_Streamlit_version

There are over 1 billion active players on Steam with over 10 thousands games available on it.
A game that other players love needs not be your cup of tea.
How do you know if a game may fit your favor and schedule without digging deep into it?

In this work, we set up a webapp pipeline that helps our user analyzing reviews of his/her game of interest. 
For this game, labelled as A, polarizied words in its reviews are extracted. 
The importance of these words are weighted by our user. 
The weightings are then used to make an estimation on our user's playtime on A using either Random Forest or Document similarity through word embeddings.
After analyzing A, games sharing similar tags with A are explored through web-scrapping contents on Steam. 
Weightings prerviously input by our user are mapped to reviews of these games. 
For each of these games, a playtime estimation is made. 
Games with a long estimated playtime are recommennded to our user.

A test run on this pipeline has been made for the game, Port Royale 4 (ID: 1024650). 
Test runs on more recently released games on Steam have also been performed during code development.
