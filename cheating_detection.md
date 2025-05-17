....

In this article we will discuss what worked, and equally importantly, what didn't and why

since this is cheating detection of a multiple choices tests, it might feel intutive to flatten the answers as a long binary vector and calculate some distance metric between each pair of students, for example,  if the choices [a,b,c,d] and the selected answer is c, it becomes [0,0,1,0] so one hot vector , and then you stack all the questions together. 

So what distance metric should we use? You might be tempted to use cosine similarity because, well, we have a bunch of vectors to compare. But while vector alignment is useful in other domains, it’s not the right tool here: the dot product doesn’t distinguish between (0,0), (1,0), or (0,1)—it’s just zeros. So, it’s natural to look for a metric that measures bit differences. If you’re an information theory enthusiast (or an unfortunate soul with an EE background), you might jump to Hamming distance!

Hamming distance counts the number of positions at which the corresponding elements are different. It seems like an elegant fit for binary answer vectors. So that’s what I tried.

But to my surprise, it turned out to be the least effective method. As you can see in the network graph, while the cheaters’ group does have a high Hamming similarity (1 - Hamming distance), so do many other pairs, just by chance. Even some honest pairs had higher similarity than the cheaters.

While it’s an elegant solution, unfortunately, the number of students means many pairs look similar by chance alone.

For a while, I tried to fix this. I thought maybe the problem was that global distance averages out important information, so I experimented with max pooling, counting exact similarities in sliding windows, and weighted averages. Nothing worked, no aggregation made a difference.

So, we have to move on. Sorry, Richard.