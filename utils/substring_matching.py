from Levenshtein import distance as ld

def get_best_match(query, corpus, case_sensitive=False, step_factor=128, favour_smallest=False):
    '''
    Returns the substrings of the corpus with the least Levenshtein distance from the queries
    (May not always return optimal answer).

    Arguments
    - query: str
    - corpus: str
    - case_sensitive: bool
    - step_factor: int  
        - Influences the resolution of the thorough search once the region is found.
          The increment in ngrams lengths used for the thorough search is calculated as ```len(query)//step_factor```.
          Increasing this increases the number of ngram lengths used in the thorough search and increases the chances 
          of getting the optimal solution at the cost of runtime and memory.
    - favour_smaller: bool  
        - Once the region of the best match is found, the search proceeds from larger to smaller ngrams or vice versa.
          If two or more ngrams have the same minimum distance then this flag controls whether the largest or smallest
          is returned.

    Returns  
    ```python
    {
        'best_match': Best matching substring of corpus,  
        'min_ld': Levenshtein distance of closest match
    }
    ```
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

    # Thorough search of the region in which the best match probably exists
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
