<font size=3>

Using Levenshtein distances to find the closest substring is a good way to find the best substring match in a corpus. Using Levenshtein distances is also much faster than `difflib`'s `ratio` function in most cases.

This code requires `python-Levenshtein`

    pip install levenshtein

The code below finds the best substring match in two stages.

The first stage constructs ngrams of the same size as the query using a sliding window over the corpus (with sliding window's step = `len(query)//2`). The ngram with the minimum Levenshtein distance from the query is found. This ngram is the region of the corpus around which the best match exists.

The second stage performs a more thorough search over the region identified (`narrowed_corpus`). This is done by constructing ngrams of varying lengths over `narrowed_corpus` with step=1. The best substring match is the ngram with the minimum Levenshtein distance from the query. The lengths of ngrams constructed depends on the argument `step_factor`. Lengths considered are

    len(query)//2 - 1, len(query)//2 - 1 + step, len(query)//2 - 1 + 2*step ... 2 * len(query) + 2

Here `step = len(query)//step_factor`. Increasing step factor increases the number of ngram lengths considered hence increasing the chances of finding the optimal substring with minimum Levenshtein distance. However increasing it will lead to increased runtime (because more ngrams need to be checked) and memory usage (more ngrams need to be stored).

```python
    from Levenshtein import distance as ld

    def get_best_match(query, corpus, case_sensitive=False, step_factor=128, favour_smallest=False):
        '''
        Returns the substring of the corpus with the least Levenshtein distance from the query
        (May not always return optimal answer).

        Arguments
        - query: str
        - corpus: str
        - case_sensitive: bool
        - step_factor: int  
            Influences the resolution of the thorough search once the general region is found.
            The increment in ngrams lengths used for the thorough search is calculated as len(query)//step_factor.
            Increasing this increases the number of ngram lengths used in the thorough search and increases the chances 
            of getting the optimal solution at the cost of runtime and memory.
        - favour_smaller: bool
            Once the region of the best match is found, the search proceeds from larger to smaller ngrams or vice versa.
            If two or more ngrams have the same minimum distance then this flag controls whether the largest or smallest
            is returned.

        Returns  
        {
            'best_match': Best matching substring of corpus,
            'min_ld': Levenshtein distance of closest match
        }
        '''

        if not case_sensitive:
            query = query.casefold()
            corpus = corpus.casefold()

        corpus_len = len(corpus)
        query_len = len(query)
        query_len_by_2 = max(query_len // 2, 1)
        query_len_by_step_factor = max(query_len // step_factor, 1)

        closest_match_idx = 0
        min_dist = ld(query, corpus[:query_len])
        # Intial search of corpus checks ngrams of the same length as the query
        # Step is half the length of the query.
        # This is found to be good enough to find the general region of the best match in the corpus
        corpus_ngrams = [corpus[i:i+query_len] for i in range(0, corpus_len-query_len+1, query_len_by_2)]
        for idx, ngram in enumerate(corpus_ngrams):
            ngram_dist = ld(ngram, query)
            if ngram_dist < min_dist:
                min_dist = ngram_dist
                closest_match_idx = idx

        closest_match_idx = closest_match_idx * query_len_by_2
        closest_match = corpus[closest_match_idx: closest_match_idx + query_len]
        left = max(closest_match_idx - query_len_by_2 - 1, 0)
        right = min((closest_match_idx+query_len-1) + query_len_by_2 + 2, corpus_len)
        narrowed_corpus = corpus[left: right]
        narrowed_corpus_len = len(narrowed_corpus)

        # Once we have the general region of the best match we do a more thorough search in the region
        # This is done by considering ngrams of various lengths in the region using a step of 1
        ngram_lens = [l for l in range(narrowed_corpus_len, query_len_by_2 - 1, -query_len_by_step_factor)]
        if favour_smallest:
            ngram_lens = reversed(ngram_lens)
        # Construct sets of ngrams where each set has ngrams of a particular length made over the region with a step of 1
        narrowed_corpus_ngrams = [
            [narrowed_corpus[i:i+ngram_len] for i in range(0, narrowed_corpus_len-ngram_len+1)] 
            for ngram_len in ngram_lens
        ]

        # Thorough search of the region in which the best match exists
        for ngram_set in narrowed_corpus_ngrams:
            for ngram in ngram_set:
                ngram_dist = ld(ngram, query)
                if ngram_dist < min_dist:
                    min_dist = ngram_dist
                    closest_match = ngram

        return {
            'best_match': closest_match,
            'min_ld': min_dist
        }

    query = "ipsum dolor"
    corpus = "lorem 1psum dlr sit amet"
    match = get_best_match(query, corpus)
    print("Best match")
    print(match['best_match'])
    print(f"Minimum Levenshtein distance: {match['min_ld']}")
```
Output

    Best match
    1psum dlr
    Minimum Levenshtein distance: 3

Input that breaks this algorithm (note the extra spaces)

    query: "antidisestablishmentarianism"
    corpus: "    anidietabishetariism    antidbmjrietabishetariism   "

Expected Output

    Best match
    anidietabishetariism
    Minimum distance: 8

Actual Output

    Best match
    antidbmjrietabishetariism
    Minimum distance: 11

</font>