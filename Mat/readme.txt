dropoutTest: 

modèle qui a donné le meilleur score sur Kaggle.
Input: embeddings de la phrase, embedding du speaker, embedding du lien qui parvient à ce noeud, embedding de la phrase précédente (embedding via bert All-mini4)

Modèle: fully connected layers avec des dropout. 
Ne semble pas apprendre énormément au cours des epochs....

Pour faire marcher le code, je sauve les embeddings avant (possible d'utiliser NN.py pour le faire)




dropoutTest+DW
pareil mais j'ai essayé d'ajouter en input les embeddings du DeepWalk de Lélia.
Donne un moins bon score



svr_withGraph...
donnait un score d'environ 0.58 sur Kaggle
input: embedding phrase, embedding speaker
modèle: SVR

Pour le faire marcher d'abord passer par training pour enregistrer le modèle puis le load dans les deux autres fichers pour la validation ou pour Kaggle

