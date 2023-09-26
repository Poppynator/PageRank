import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1]) 
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    dist = {}
    
    if page in corpus and len(corpus[page]) > 0:
        # Distribute the damping factor equally among linked pages
        linked_prob = damping_factor / len(corpus[page])
        for linked_page in corpus[page]:
            dist[linked_page] = linked_prob

    # Distribute the remaining probability uniformly to all pages
    uniform_prob = (1 - damping_factor) / num_pages
    for other_page in corpus:
        dist.setdefault(other_page, 0)
        dist[other_page] += uniform_prob

    return dist
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        transition_probs = transition_model(corpus, page, damping_factor)
        page = random.choices(list(transition_probs.keys()), weights=transition_probs.values(), k=1)[0]

        # Update the PageRank for the chosen page
        page_rank[page] += 1

    # Normalize PageRank values to sum to 1
    total_rank = sum(page_rank.values())
    page_rank = {page: rank / total_rank for page, rank in page_rank.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    num_pages = len(corpus)
    initial_rank = 1 / num_pages
    ranks = {page: initial_rank for page in corpus}
    threshold = 0.0005

    while True:
        new_ranks = {}
        total_diff = 0

        for page in corpus:
            new_rank = (1 - damping_factor) / num_pages
            link_sum = sum((ranks[linked_page] / len(corpus[linked_page])) for linked_page in corpus if page in corpus[linked_page])
            new_rank += damping_factor * link_sum
            new_ranks[page] = new_rank

            total_diff += abs(new_ranks[page] - ranks[page])

        if total_diff < threshold:
            break

        ranks = new_ranks

    return ranks

if __name__ == "__main__":
    main()
