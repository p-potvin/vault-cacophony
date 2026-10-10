# French/English 11.6 min — whole-file vs VAD-segmented (and fr-CA)

Fixed build, all decoded offline.

| condition | words | gaps | dropped |
|---|---|---|---|
| whole-file `--language auto` | 2348 | 7 | 23.3 s |
| VAD-segmented (7 seg), `auto` | 2363 | 4 | 12.2 s (5.7 s warmup) |
| whole-file `--language fr-CA` | 1621 | 21 | 185.5 s |

## Time-aligned (15 s buckets)

**00:00**

- whole/auto: Can a fringe person understand a Quebec French speaker from Canada? Apparently not according to the many comments on my last video, I'm French. I live in Paris and today I want to react to a few videos of people speaking Quebec French and see how much of it I can actually understand.
- VADseg    : Can a fringe person understand a Quebec French speaker from Canada? Apparently not according to the many comments on my last video, I'm French. I live in Paris and today I want to react to a few videos of people speaking Quebec French and see how much of it I can actually understand.
- fr-CA     : Quebec French and see how much of it I can actually understand.

**00:15**

- whole/auto: So if you've never heard Quebec French before, it is pretty different from the French spoken in France and you need to know that French is the only official language of Quebec in Canada and it's because back in the seventeenth century Canada was a French colony until it was lost to
- VADseg    : So if you've never heard Quebec French before, it is pretty different from the French spoken in France and you need to know that French is the only official language of Quebec in Canada and it's because back in the seventeenth century Canada was a French colony until it was lost to
- fr-CA     : So, if you've never heard Quebec French before, it is pretty different from the French spoken in France, and you need to know that French is the only official language of Quebec in Canada, and it's because back in the seventeenth century Canada was a French colony until it was lost to

**00:30**

- whole/auto: Britain in seventeen sixty three. French speakers there became surrounded with English speakers, but they reheld on tightly to their culture and their language, which evolved totally separately from the French from France, and as a French person, one of the questions I had when researching this
- VADseg    : Britain in seventeen sixty three. French speakers there became surrounded with English speakers, but they reheld on tightly to their culture and their language, which evolved totally separately from the French from France, and as a French person, one of the questions I had when researching this
- fr-CA     : Britain in seventeen sixty three. French speakers later became surrounded with English speakers, but they reheld on tightly to their culture and their language, which evolved totally separately from the French from France, and as a French person, one of the questions I had when researching this

**00:45**

- whole/auto: video is how is Quebec keeping their language alive when there are so many English speakers around them and to understand that, let's watch our first video tous les jeunes adultes que nous interpellons ici à Montréal intègrent l'anglais dans leur vie courant. Moi, j'ai que toutes mes séries en anglais,
- VADseg    : video is how is Quebec keeping their language alive when there are so many English speakers around them and to understand that, let's watch our first video tous les jeunes adultes que nous interpellons ici à Montréal intègrent l'anglais dans leur vie courant. Moi, j'ai que toutes mes séries en anglais,
- fr-CA     : video is how is Quebec keeping their language alive, there are so many English speakers around them, and to understand that, let's watch our first video. Tous les jeunes adultes que nous interpellons ici à Montréal intègrent l'anglais dans leur vie courant. Moi, j'ai que toutes mes séries en anglais,

**01:00**

- whole/auto **[gap]**: même dans ma tête, quand je parle avec quelqu'un, je vais penser en anglais there's over eight million people in the province of Quebec, so it's one of the most well known ways of speaking French. Everyone who just spoke here has a very understandable accent
- VADseg     **[gap]**: même dans ma tête, quand je parle avec quelqu'un, je vais penser en anglais there's over eight million people in the province of Quebec, so it's one of the most well known ways of speaking French. Everyone who just spoke here has a very understandable accent
- fr-CA      **[gap]**: même dans ma tête, quand je parle avec quelqu'un, je vais penser en anglais. First of all, I want to say that I understand everyone very well eight million people in the province of Quebec, so it's one of the most well known ways of speaking French.

**01:15**

- whole/auto: to me as a French person, even though you'll see there's quite a few differences selon un rapport de le fils québécois de la langue française, le tiers des Québécois de dix huit à trente quatre ans préfère travailler dans un environnement bilingue à peine cinquante huit pour cent travaillent essentiellement en
- VADseg    : to me as a French person, even though you'll see there's quite a few differences selon un rapport de le fils québécois de la langue française, le tiers des Québécois de dix huit à trente quatre ans préfère travailler dans un environnement bilingue à peine cinquante huit pour cent travaillent essentiellement en
- fr-CA     : Selon un rapport de l'office québécois de la langue française, le tiers des Québécois de dix huit à trente quatre ans préfère travailler dans un environnement bilingue. À peine cinquante huit pour cent travaillent essentiellement en

**01:30**

- whole/auto **[gap]**: français en diminution constante depuis deux mille dix. Ça me dérange pas du tout. More and more people have to speak English at work,
- VADseg    : français. En diminution constante depuis deux mille dix, faut pas qu'on perd le Français non plus, c'est sûr, mais les deux langues, ça me dérange pas du tout. So that's really interesting to me that especially in Montréal more and more people have to speak English at work
- fr-CA      **[gap]**: français en diminution constante depuis deux mille dix. C'est sûr, mais les deux langues, ça me dérange pas du tout.

**01:45**

- whole/auto: and I have found in my research that there's a lot of talk on how to keep the French language alive, and you know I also relate to what this girl said because it's not about refusing English, but it's more about keeping your culture, your language, and your identity à Montréal où les francophones
- VADseg    : and I have found in my research that there's a lot of talk on how to keep the French language alive and you know I also re relate to what this girl said because it's not about refusing English but it's more about keeping your culture, your language, and your identity à Montréal où les francophones
- fr-CA     : Your langue, your identity à Montréal où les francophones

**02:00**

- whole/auto: représentent soixante six pour cent de la population, alors que la proportion se situe à quatre vingt quatorze pour cent ailleurs en province. It feels like there's a big gap between Montreal and the rest of the province like sixty six percent versus eighty four percent is quite
- VADseg    : représentent soixante six pour cent de la population, alors que la proportion se situe à quatre vingt quatorze pour cent ailleurs en province. It feels like there's a big gap between Montreal and the rest of the province like sixty six percent versus eighty four percent is quite
- fr-CA      **[gap]**: représentent soixante six pour cent de la population alors que la proportion se situe à quatre vingt quatorze pour cent ailleurs en province. Like sixty six percent vers eighty four percent, is quite

**02:15**

- whole/auto **[gap]**: a big gap, and since we're on the topic of the rest of Quebec, let's now make it harder for me and let's try to understand a Quebec accent of someone who's not from Montreal. On peut bien direment on le sait pour personne ne parler avec un roman soit
- VADseg     **[gap]**: a big gap and since we're on the topic of the rest of Quebec, let's now make it harder for me and let's try to understand a Quebec accent of someone who's not from Montreal. On peut bien dire qu'il y a du roman où le sait pour personne va parler avec un roman soit
- fr-CA      **[gap]**: a big gap, en since on the topic of the rest of Quebec, let's now make it harder for me and let's try to understand a Quebec accent of someone who is not from Montreal. On peut bien dire il y a du roman où le sait pour personne ne parler avec un roman,

**02:30**

- whole/auto **[gap]**: very hard for me I get the context you know his uh fisherman out of the beautiful île de la Madeleine but I don't understand too much so let's put subtitles on ballen o kui vomete HPF me o molo, set a dû à suivre
- VADseg    : this is very hard for me I get the context you know his uh fisherman out of the beautiful île de la Madeleine but I don't understand too much so let's put subtitles on C'est pas comme une baleine ou cuisonmente a GPS, mais agomolo, c'est très lieu
- fr-CA      **[gap]**: I get the contexterman out of the beautiful île de la Madeleine, but I don't understand too much, so let's put subtitles on Buite HPF me molo, c'est créadu à suivre.

**02:45**

- whole/auto: ti a bien liste qui continue n'importe quoi, mais n'a pas en deux autres qui a pas oublié avec un roman quelqu'un parlé avec un roman et voir a trouvé la recette Maique. It is tough, but I feel if I train my ears I would a hundred percent understand him and I don't know about
- VADseg    : à suivre, puis y a bien les billeries qui comptent n'importe quoi, mais la pendule d'autres, qui a pas volé avec un roman quelqu'un a volé avec un roman, il va trouver la recette ma heure. It is tough, but I feel if I train my ears I would a hundred percent understand him and I don't know about
- fr-CA      **[gap]**: Quelqu'un parlé avec un roman et voir a trouvé la recette Maique. It is tough, but I feel if I train my ears a hundred percent understand him, and I don't know about

**03:00**

- whole/auto: you but I feel he's accent reminds me of the Cajun French speakers from Louisiana, which I already made a video about, and it's so interesting to me to see how many different accents there are and just know as well that there's so many different French speakers in Canada. Today we're gonna focus on
- VADseg    : you but I feel he's accent reminds me of the Cajun French speakers from Louisiana, which I already made a video about, and it's so interesting to me to see how many different accidents there are and just know as well that there's so many different French speakers in Canada today we're gonna focus on
- fr-CA     : you, but I feel he accent reminds me of the Cajun French speakers from Louisiana, which I already made a video about, and it's so interesting to me to see how many different accents there are and just know as well that there's so many different French speakers in Canada Today we're gonna focus on

**03:15**

- whole/auto: Quebec French, but if you like this video, I might make more. Also, I just feel I need to share this, but every person from Quebec that I've met told me that French people from France often found their accent funny, which seriously makes me feel a bit sad and disappointed because Quebec French and other French
- VADseg    : Quebec French, but if you like this video, I might make more. Also, I just feel I need to share this, but every person from Quebec that I've met told me that French people from France often found their accent funny which seriously makes me feel a bit sad and disappointed because Quebec French and other French
- fr-CA      **[gap]**: Quebec French, but if you like this video, I might make more.

**03:30**

- whole/auto: accents are a hundred percent valid ways of speaking French to me and also I want to add that it's not just these people some French people love making fun of accents like I personally remember when I moved to Paris from the South of France all these years ago there were a few Parisians who made fun of
- VADseg    : accents are a hundred percent valid ways of speaking French to me and also I want to add that it's not just these people some French people love making fun of accents like I personally remember when I moved to Paris from the south of France all these years ago there were a few Parisians who made fun of
- fr-CA     : France all de years ago, Parisians, made fun of

**03:45**

- whole/auto: me for having a Southern accent like I remember I couldn't say a word with having those people laughing at me and I think even fifteen years ago I'm still a bit salty about it. I just wish people would approach differences with a bit more curiosity because it just makes life way more interesting. All right, one thing
- VADseg    : me for having a southern accent like I remember I couldn't say a word with having those people laughing at me and I think even fifteen years ago I'm still a bit salty about it. I just wish people would approach differences with a bit more curiosity because it just makes life way more interesting. All right, one thing I
- fr-CA      **[gap]**: me for having a southern accent, I remember I couldn't say a word with having those people laughing at me, and I think even fifteen years ago I'm still a bit salty about it. One thing

**04:00**

- whole/auto: I did know about Quebec is that they have so many expressions that are different than the ones we would use in France. So let's check them out a Quebec, on se peux se balader, on y prendre une marche d'inference, this would mean like you fell au Québec on second ou il fait froid,
- VADseg    : did know about Quebec is that they have so many expressions that are different than the ones we would use in France. So let's check them out au Québec, on se peux se balader. On dit prendre une marche infrance, this would mean like you fell au Québec, on se peut ou il fait
- fr-CA     : I didn't know about Quebec is that they have so many expressions that are different than the ones we would use in France, so let's check them out au Quebec, on se peux se balader, on y prendre une marche d'inference, this would mean au Québec on se sait pas où il fait froid,

**04:15**

- whole/auto: on dit il fait fret ou on gèle au Québec, on fait pas du shopping en magazine au Québec un petit peu bonjour en Slovain, on dit bon matin, yeah I knew this one as well. There's a few that kind of mean something
- VADseg    : froid, on dit il fait fret où on gèle au Québec, on fait pas du shopping, en magazine au Québec est un petit peu bonjour en Slevan, on dit bon matin. Yeah, I knew this one as well. There's a few that. Kind of mean something different
- fr-CA      **[gap]**: on dit il fait fret ou on gèle au Québec, on fait pas du shopping en magazine au Québec un petit peu bonjour en Slovain, on dit bon matin, y a I knew this one as well.

**04:30**

- whole/auto: different in France, but I think I would understand in context. Also I love magazine because in French we would use a mixture of French and English like we would say faire du shopping with a French accent and unlike Quebec in France we're really famous for our very
- VADseg    : in France, but I think I would understand in context. Also, I love magaziner because in French we would use a mixture of French and English like we would say faire du shopping with a French accent and unlike Quebec in France we're really famous for our
- fr-CA     : —

**04:45**

- whole/auto: weird use of English like instead of translating movie titles, we give them a weird English title and then we say it with a big French accent like in France this movie is not the hangover is very bad trip but in Quebec it's lendement devey, which sounds
- VADseg    : very weird use of English like instead of translating movie titles, we give them a weird English title and then we say it with a big French accent like in France this movie is not the hangover is very bad trip, but in Quebec it's lendement de Vay, which sounds
- fr-CA     : It's very bad trip but in Quebec it's lendement devey, which sounds

**05:00**

- whole/auto: so much nicer in my opinion and Quebec seems to translate so many more things into French, which in my personal opinion is really nice because I cannot stand the bad Franklish in France anymore, especially in ads like seriously they can be ridiculous sometimes in any case
- VADseg    : so much nicer in my opinion, and Quebec seems to translate so many more things into French, which in my personal opinion is really nice because I cannot stand the bad Franklish in France anymore, especially in ads like seriously they can be ridiculous sometimes in any case
- fr-CA     : so much nicer in my opinion, and Quebec himself translate so many more things into French, which in my personal opinion is really nice because I cannot stand the bad Franglish in France anymore, especially in ads, like seriously, they can be ridiculous sometimes, in any case,

**05:15**

- whole/auto: I really love all these differences if you're going to travel abroad soon, you need to hear about today's sponsor Alo with Airlow you can buy affordable data plans around the world through ESIMs in over two hundred countries and regions I've been using them for over two years
- VADseg    : I really love all this differences if you're going to travel abroad soon, you need to hear about today's sponsor Alow with Arlow, you can buy affordable data plans around the world through ESIMS in over two hundred countries and regions. I've been using them for over two
- fr-CA     : I really love all this differences, If you're going to travel abroad soon, you need to hear about today's sponsor Airlo with Airlow, you can buy affordable data plans around the world through ESIMs in over two hundred countries and regions. I've been using them for over two years,

**05:30**

- whole/auto: and they are my favorite travel hack to have data on my phone when I land. I recently had to go to the Philippines for a wedding and I installed my ESIM on my phone before getting on the plane and when I landed I just had to activate it to have data on my phone didn't have to worry about finding a
- VADseg    : years and they are my favorite travel hack to have data on my phone when I land. I recently had to go to the Philippines for a wedding, and I installed my ESIM on my phone before getting on the plane, and when I landed, I just had to activate it to have data on my phone, didn't have to worry about finding a
- fr-CA     : and they are my favorite travel hack to have data on my phone when I land. I recently go to the Philippines for a wedding and I installed my ESIM on my phone before getting on the plane, and when I land it, I just had to activate it to have data on my phone didn't have to worry about finding a

**05:45**

- whole/auto: Wi Fi connection or looking for a physical Sim card. One thing I really like about Arrow is that they have so many different lookations available, so it really adds peace of mind for me like I know I have one less thing to worry about when I'm traveling you can even use my code
- VADseg    : Wi Fi connection or looking for a physical SIM card. One thing I really like about Arrow is that they have so many different locations available, so it really adds peace of mind for me. Like I know I have one less thing to worry about when I'm traveling. You can even use my code Lucille
- fr-CA      **[gap]**: Wi Fi connection or looking for a physical Sim card. Aerrow is that they have so many different locations available, so it reads peace of mind for me, I know I have one less thing to worry about when I'm traveling you can even use my code

**06:00**

- whole/auto: Lucille three to get three dollars off your first ESIM they also have regional plans as well. You can buy their Europe ESIM and get data in forty two countries for the duration of your trip if you want to try it, you can download the aerolog using the link in the description or
- VADseg    : three to get three dollars off your first ECIN. They also have regional plans as well. You can buy their Europe ESIM and get data in forty two countries for the duration of your trip if you want to try it, you can download the aero app using the link in the description or scan
- fr-CA      **[gap]**: Lucille to get three dollars off your first ESIM. Quet data in forty two countries for the duration of your trip, you want to try it, you can download the aerologue app using the link in description or

**06:15**

- whole/auto: scan the QR code and use my code Lucille. So one thing I knew about Quebec is that they regul the extra mile to promote the French language and sometimes it's done in ways not everyone agrees with and we will not re get into that today because this is supposed to be a fun
- VADseg    : the QR code and use my code Lucile Three. So one thing I knew about Quebec is that they regold the extra mile to promote the French language, and sometimes it's done in ways not everyone agrees with, and we will not really get into that today because this is supposed to be a fun and light
- fr-CA      **[gap]**: scan the QR code and use my code Lucille.

**06:30**

- whole/auto: and light discovery of the language and the culture, but one way I refound interesting though is the French classes that are offered to new immigrants who don't yet speak French, and here I found this amazing documentary that follows a group of people from all over the world who just arrived
- VADseg    : discovery of the language and the culture. But one way I refound interesting though is the French classes that are offered to new immigrants who don't yet speak French, and here I found this amazing documentary that follows a group of people from all over the world who just arrived recently
- fr-CA     : —

**06:45**

- whole/auto: recently in Quebec and are trying to learn French in six months, which can help them with job opportunities or getting permanent residency. À quoi on s'attendre ces étudiants là, les étudiants qui finissent le cours deux vont être capables de raconter des expériences personnelles
- VADseg    : in Quebec and are trying to learn French in six months, which can help them with Java opportunities or getting permanent residency à quoi s'attendre ces étudiants là les étudiants qui finissent le cours deux vont être capables de raconter des expériences personnelles
- fr-CA     : À quoi on s'attendre ces étudiants là, les étudiants qui finissent le cours deux vont être capables de raconter des expériences personnelles

**07:00**

- whole/auto: sont évidemment capables d'aller au restaurant, magasiner, s'informer avec un propriétaire. Okay, so what I love about this class is that it's very concrete, like it's not just grab our and vocab, it's all the situations you would encounter in your daily life if you had just moved
- VADseg    : sont évidemment capables d'aller au restaurant, magasiner, s'informer avec un propriétaire. Okay, so what I love about this class is that it's very concrete, like it's not just grammar and vocab it's all the situations you would encounter in your daily life if you had just moved
- fr-CA     : sont évidemment capables d'aller au restaurant, magasiner, s'informer avec un propriétaire. Okay, soit I love about this class is that it's very concrete, it's not just graver and vocab it's all the situations you would encounter in your daily life

**07:15**

- whole/auto **[gap]**: to a country. So like going to the restaurant, finding an apartment, finding work, and that's so important because moving to a new place when you don't speak the language can be very intimidating comme adulte de faire des erreurs,
- VADseg     **[gap]**: to a country so like going to the restaurant, finding an apartment, finding work, and that's so important because moving to a new place when you don't speak the language can be very intimidating comme adulte. faire
- fr-CA      **[gap]**: to a country, so like going to the restaurant finding an apartment, finding work, and that's so important because moving to a new place, you don't speak the language can be very intimidating comme adult de faire des erreurs,

**07:30**

- whole/auto: de pas produire des phrases aussi claires que ce qu'on a dans la tête. This teacher seems awesome. I don't want to spoil the ending of the documentary, but he's doing such a good job at preparing them for everyday situations, which is truly what you need if you're learning French
- VADseg    : des erreurs, de pas produire des phrases aussi claires que ce qu'on a dans la tête. This teacher seems awesome. I don't want to spoil the ending of the documentary, but he's doing such a good job at preparing them for everyday situations, which is truly what you need if you're learning
- fr-CA      **[gap]**: de pas produire des phrases aussi claires que ce qu'on a dans la tête. This teacher ending of the documentary, but he's doing such a good job at preparing for everyday situations, which is truly what you need if you're learning French

**07:45**

- whole/auto: in six months instead of focusing on trying to be perfect, and I found that so interesting that there's different programs to help people learn the language when they come to Quebec, and they also have official programs for people who already are French speakers like in France there's often some official meetings that are
- VADseg    : French in six months instead of focusing on trying to be perfect, and I found that so interesting that there's different programs to help people learn the language when they come to Quebec, and they also have official programs for people who already are French speakers. Like in France, there's often some official meetings that are organized
- fr-CA      **[gap]**: in six months instead of focusing on trying to be perfect. Like in France, often official meetings that are

**08:00**

- whole/auto: organized by Quebec to promote French immigration to Quebec, and I know that there's a lot of young French people who want to come to Quebec, Montreal in particular, because they make it so easy for French people to come and go there with even having a specific type of visa which makes it
- VADseg    : by Quebec to promote French immigration to Quebec. And I know that there's a lot of young French people who want to come to Quebec, Montreal in particular because they make it so easy for French people to come and go there with even having a specific type of visa, which makes it
- fr-CA     : organized by Quebec to promote French immigration to Quebec, and I know that there's a lot of young French people Quebec, Montreal in particular, because they make it so easy for French people to come and go there with even having a specific type of visa which makes it

**08:15**

- whole/auto: easier for French speakers to come to Quebec. I personally know a lot of French people who have spent at least six months in Quebec either for work, for an internship, for their studies, and it's often seen by French people as an easier way to discover a new culture
- VADseg    : easier for French speakers to come to Quebec. I personally know a lot of French people who have spent at least six months in Quebec either for work for an internship for their studies and it's often seen by French people as an easier way to discover a new culture since
- fr-CA     : easier for French speakers to come to Quebec. I personally know a lot of French people have spent at least six months in Quebec either for work for an internship for their studies, and it's often seen by French people as an easier way to discover a new culture

**08:30**

- whole/auto: since we speak the same language, but it's not always easy as we're gonna see in this next video, which is a French documentary about French people who are coming to Quebec in search of good job opportunities, and here we're following this guy who is applying to different positions in
- VADseg    : we speak the same language, but it's not always easy as we're going to see in this next video, which is a French documentary about French people who are coming to Quebec in search of good job opportunities, and here we're following this guy who is applying to different positions in cafes
- fr-CA     : since we speak the same language, but it's not always easy as we're gonna see in this next video, which is a French documentary about French people Quebec in search of good job opportunities, and here we're following this guy, to different positions in

**08:45**

- whole/auto: cafes and receives feedback on cultural differences and how to find work. Aucun poste dans l'immédiat, mais la gérante prend le temps de conseiller Gaétan sur son C V pas tout à fait adapté aux exigences locales, t'as même pas écrit jeu de société.
- VADseg    : and receives feedback on cultural differences and how to find work. Aucun poste dans l'immédiat. Mais la gérante prend le temps de conseiller Gaétan sur son C V, pas tout à fait adapté aux exigences locales.
- fr-CA      **[gap]**: cafes and receives feedback on cultural differences and how to find work. Aucun poste dans l'immédiat, mais la gérante prend le temps de conseiller Gaétan sur son CV, pas tout à fait adapté aux exigences locales.

**09:00**

- whole/auto: Oh ouais, puis essaye de pas utiliser des termes anglo français commercials, c'est pas un mot ici, ça explique, ça veut rien dire. So this is so interesting that he's receiving that as a feedback because there are so many English words at work in French
- VADseg    : Oh ouais, puis essayer de pas utiliser des termes anglo français commerciales, c'est pas un mot ici, ça existe pas, ça veut rien dire Okay, so this is so interesting that he's receiving that as a feedback because there's so many English words ad work in French
- fr-CA      **[gap]**: Oh ouais, puis essaye de pas utiliser des termes anglo français commercials, c'est pas un mot ici, ça explique, ça veut rien dire.

**09:15**

- whole/auto: like we say meeting, email, customer success, sales all of these words we say them with the French Jackson. All right, let's keep going ouais en France, c'est vraiment buy the booklet vraiment très propre, tout seul et j'arrive leur photo sur leur C V, ils sont vraiment très carrés. Nous, on va regarder
- VADseg    : like we said meeting, email customer success, sales all of these words we say them with the French accent. All right, let's keep going ouais en France, c'est vraiment buy the book, le vraiment très propre, seul et j'arrive mettre leur photo sur leur C V, ils sont vraiment très carrés. Nous, on va regarder
- fr-CA     : Success, sales all of those words we sam with a French Jackson. Allait, let's keep going ouais en France, c'est vraiment buy the book là vraiment très propre, tout seul et j'arrive leur photo sur leur C V. Ils sont vraiment très carrés. Nous on va regarder

**09:30**

- whole/auto: déjà l'énergie déjà qui est importante puis après où à l'expérience, ça t'a pas mieux que ça, t'es allé à l'école cinq ans en photographie, ça veut pas dire que tu fais de la photo et it seems to be quite a big cultural difference. It's true that in France as a society we're still very formal we value education
- VADseg    : déjà l'énergie déjà qui est importante puis après où elle l'expérience, ça t'a pas mieux que ça, c'était allé à l'école cinq ans en photographie, ça veut pas dire que tu fais de la photo. It seems to be quite a big cultural difference. It's true that in France as a society we're still very formal. We value education
- fr-CA     : déjà l'énergie déjà qui est importante puis après où à l'expérience, ça t'a pas mieux que ça, t'es allé à l'école cinq ans en photographie, ça veut pas dire que tu fais de la photo et it seems toi difference et true in France, as a society, we're still very formal we value education

**09:45**

- whole/auto: a lot and here it seems clear that at least for this guy there seems to be a difference in culture and even if you speak the same language you still need to adapt it's not because you can understand what someone says that you don't need to make an effort to adapt to their culture when you're coming
- VADseg    : a lot and here it seems clear that at least for this guy there seems to be a difference in culture, and even if you speak the same language, you still need to adapt. It's not because you can understand what someone says that you don't need to make an effort to adapt to their culture when you're coming
- fr-CA     : a lot, and here it seems clear that at least for this guy there seems to be a difference in culture, and even if you speak the same language, you still need to adapt. It's not because you can understand what someone says, that you don't need to make an effort to adapt to their culture when you're coming

**10:00**

- whole/auto: to their country all right now let's see it from a Quebec perspective and watch this video by a French creator that I love who went around and asked people from Quebec what they thought of French people parle Parisian all say que c'est pas les plus akéans,
- VADseg    : to their country. All right, now let's see it from a Quebec perspective and watch this video by a French creator that I love who went around and asked people from Quebec what they thought of French people Parisian,
- fr-CA      **[gap]**: to their country. Parlia Parisien, c'est pas les plus acaïens,

**10:15**

- whole/auto **[gap]**: c'est pas les plus chaleureux, t'es une relation un peu compliquée, on a de la difficulté à se comprendre, on a bien beau parler la même langue, on s'exprime pas toujours de la même façon français en général, on vous appelait comptable que je connais pas les bonnes personnes
- VADseg    : c'est pas les plus chaleureux, t'as une relation un peu compliquée, on a de la difficulté à se comprendre, on a bien vous parlé la même langue, on s'exprime pas toujours de la même façon français en général, on vous apprend, on est comptable.
- fr-CA      **[gap]**: c'est pas les plus chaleureux, t'as une relation un peu compliquée. On a de la difficulté à se comprendre, on a bien beau parler la même langue, on s'exprime pas toujours de la même façon français en général, on vous appelle que je connais pas les bonnes personnes

**10:30**

- whole/auto: très lucides sur la réalité très politique, très critique this is so true. French people love to complain and critique stuff that's another thing they like I saw a good disagreeable, parfois unpur la vie est généralement positive, sauf que
- VADseg    : très lucide sur la réalité, très politique, très critique. This is so true French people love to complain and critique stuff. That's another thing I like. I saw a good disagreeable parfois unpush la vie est généralement positif, sauf que des
- fr-CA      **[gap]**: très lucides sur la réalité très politique, très critique. French people love to complain, parfois un peu snubbie est généralement positive, sauf que

**10:45**

- whole/auto **[gap]**: des fois on trouve que vous énervez un peu facilement I feel personally attacked by that on les aime beaucoup vous êtes nos cousins, yeah that's also a thing French people I don't know if that's the case if you're from Quebec let me know you say we are cousins as well we always
- VADseg     **[gap]**: fois on trouve que vous énervez un peu facilement attacked by that on les aime beaucoup, vous êtes nos cousins, yeah that's also a thing a French people I don't know if that's the case if you're from Quebec, let me know. Do you say we're cousins as well we always
- fr-CA      **[gap]**: des fois on trouve que vous énervez un peu facilement I feel personnally attacked by that on lesme beaucoup vous êtes nos cousins French people I don't know that's the case, from Quebec, let me know you say we are cousins as well, we always

**11:00**

- whole/auto: say that Quebec people are cousins we say no cousin québécois so let me know if you're from Quebec are we are we cousins but I do feel this way quand même très sophistiqué on aime beaucoup votre cuisine, vous êtes quand même très accueillant, sauf
- VADseg    : say that Quebec people are cousins we say no cousin québécois so let me know if you're from Quebec are we cousins but I do feel this way quand même très sophisqué, on aime beaucoup votre cuisine, vous êtes quand même très accueillant, sauf des
- fr-CA     : say that Quebec people are cousins we say no cousin québécois, so let me know if you're from Quebec, or we are we cousins, but I do feel this way quand même très sophistiqué, on aime beaucoup votre cuisine, vous êtes quand même très accueillant, sauf

**11:15**

- whole/auto: des papas, ils rigole beaucoup notre accent qu'on ne peut d'ailleurs non, sont des gens sympathiques qui sont très curieux et qui ont souvent une très grande estime de même loved those rapid fire questions they were so funny and you know I think it's the
- VADseg    : contains, ils rigole beaucoup notre accent qu'on ne peut d'ailleurs, non, sont des gens sympathiques qui sont très curieux et qui ont souvent une très grande estime de main loved those rapid fire questions they were so funny and you know I think it's the perfect
- fr-CA     : des papas, ils rigole beaucoup notre accent qu'on ne peut d'ailleurs n'en sont des gens sympathiques qui sont très curieux et qui ont souvent une très grande estime de même loved those rapid fire questions.

**11:30**

- whole/auto: perfect way to end that video it was so fun to discover the language and the culture of Quebec and now I just need to travel there.
- VADseg    : way to end that video it was so fun to discover the language and the culture of Quebec and now I just need to travel there.
- fr-CA     : —

