###################################
##
# Valon
# 11-16-2025
## Objective: This code takes the list of slang words and compares against a comprehensive sample of slang words
#NOTE: This slang dictionary is obtained from chatgpt scouring words recently added to Merriam Webster, Dictionary.com
# and other proiminent dictionaries. Additional work could be done by scouring top words on urban dictionary.
###################################

import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

slang_df = pd.read_csv('output1/slang_weights_top_vocab.csv')
       
SPACY_BATCH = 2000  # tokens processed per nlp.pipe batch
NLP_N_PROCESS = 1   # set >1 if your spaCy supports multiprocessing on your machine
MIN_TOKEN_LEN = 2 # avoid singular chars

new_words_sample = [
"rizz","zhuzh","doggo","padawan","simp","goated","bussin","cromulent","mid","ngl",
"tfw","ttyl","burrata","capicola","freestyle","burrito","cakeage","jawn",
"nepo","algo","informationpollution","petfluencer","jolabokaflod","kakeibo","decisionfatigue",
"showerorange","liminalspace","hellscape","selfcoup","ragefarming","northpaw","nearlywed",
"skibidi","tradwife","delulu","yeet","nibling","floof","grammable","boujee","amirite",
"bacne","bedrotting","girlDinner","bedrotting","brainrot","mousejiggler","foreverchemical",
"touchgrass","fyp","fyp (FYP abbreviation)","forYoupage","rizzled","chefskiss","boop","theick",
"porchpiracy","speedrun","sidequest","ultraprocessed","heatindex","spottedlanternfly","shadowban",
"idgaf","bingo","bingoCard","panicdemic","cryptojacking","llm","gpt","hallucinate (ai sense)",
"petfluencer","bussin","cashgrab","creepycrawly","maga","classicalLiberalism","latecapitalism",
"skibidi","tradwife","delulu","mousejiggler","skibidi","skibidi","selfiecore","barbiecore",
"girlDinner","bedrotting","brainrot","bop","moodboard","tokumentary","microinfluencer",
"microcheating","doomscrolling","doomscroll","finfluencer","finfluencing","sponcon","adtech",
"cryptomining","metaverse","web3","deplatform","crowdsourcing","crowdfund","clickbait",
"infodemic","infodemic","covidiot","doomscrolling","doomscroll","deepfake","deepfakes","deepfakeing",
"autofill","autofillings","airdrop","airdrops","NFT","nft","nfting","metaverse","metaverses",
"infrared","antisemitism","latine","talmbout","petfluencer","nepoBaby","nepo","nibling",
"skibidi","grammable","yeeted","yeeting","yeetage","cancelling","cancelculture","deplatforming",
"latinx","latine","pansexual","nonbinary","xennial","okboomer","glowup","glowups","stan","staning",
"stanTwitter","stanAccount","simping","simped","simp","soggy","spoilery","snackable","seacore",
"rando","randoing","zesty","zestiness","bet","betting","beton","wokeadj","woke","wokeish","cryptobro",
"cryptobroish","memeing","memeworthy","memeable","goblinmode","goblinmodey", "nepo baby", "idgaf", "girl dinner", "bed rotting", "enshittification",
    "barbiecore", "beach read", "the ick", "bussin"]

# tokenize word sample
docs = nlp.pipe(new_words_sample, batch_size=SPACY_BATCH, n_process=NLP_N_PROCESS)
tokens = [[tok.lemma_.lower() for tok in doc  if tok.is_alpha and len(tok.text) >= MIN_TOKEN_LEN] for doc in docs]
token_set = list(set(tokens[0]))
# accuracy test:
recall = sum([1 if word in token_set else 0 for word in slang_df['SlangWord']])/len(token_set)
print(len(slang_df['SlangWord']))
print(f'Recall: {recall:.2f}')

            


