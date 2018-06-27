#!/usr/bin/env python3

#
# cluster-benchmark.py
#   Copyright (C) 2018 Acquire Media / Newscycle Solutions
#
# 2018-05-08 Jonathan A. Marshall
#
# Python program to run benchmark tests comparing a MIMOSA clustering
#   algorithm to a Centroid clustering algorithm.  See the associated
#   research paper for details.
#

import abc        # For abstract base classes.
import sys        # For commandline args and I/O.
import re         # Regular expressions.
import time       # For benchmarking time.
import resource   # For benchmarking memory usage.
import itertools  # For combinatorics.


class DataCluster (abc.ABC):
    """Subclass must implement cluster() method."""

    @abc.abstractmethod
    def cluster (self, sig_num, sim_threshold, sizes, elements):
        """The cluster() method must be implemented by a subclass."""
        pass


class DataClusterLinear (DataCluster, abc.ABC):
    """Implement a MIMOSA linear-time clustering algorithm."""

    @abc.abstractmethod
    def similarity_size (self, mi_size, mo_size, ov_size):
        """The similarity_size() method must be implemented by a subclass.

        Arguments:
        mi_size -- Number of elements in one of the sets.
        mo_size -- Number of elements in the other set.
        ov_size -- Number of elements in the overlap (intersection) of
                   the two sets."""
        pass

    def __init__ (self, args):
        """Populate data needed for MIMOSA clustering.

        Arguments:
        args -- A dictionary whose keys are to become attributes of
                the DataClusterLinear object.  Required keys are:
                sim_threshold, sizes."""
        if not args.get('sizes'): raise ValueError('Sizes must be provided')
        for key in args.keys(): setattr(self, key, args[key])
        self.markers = {}

        # Invoke helper function to construct the MatchOut and MarkIn tables.
        self.mi_table, self.mo_table = self.make_mimo_table(self.sim_threshold,
                                                            self.sizes)

        # Precompute all partial signature combinations for each possible
        #   overlap size around signatures of each allowable size.
        self.combinations = [[] for sig_size in range(self.sizes[-1] + 1)]
        for sig_size in self.sizes:
            for ov_size in range(self.mi_table[sig_size][-1] + 1):
                self.combinations[sig_size].append([])
        for sig_size in self.sizes:                   # Allowable signature sizes.

            nums = list(range(sig_size))

            for ov_size in self.mi_table[sig_size]:   # Overlap sizes.

                if ov_size > sig_size: continue       # Skip overlaps that are too big.

                # Store index values for the combination elements into a table.
                self.combinations[sig_size][ov_size] = list(
                    itertools.combinations(nums, ov_size))

    def make_mimo_table (self, sim_threshold, sizes):
        """Construct and initialize MatchOut and MarkIn tables.

        Arguments:
        sim_threshold -- Similarity threshold value between 0 and 1.
        sizes -- List of allowable sizes of input signatures."""
        mark_in_table   = [[] for i in range(sizes[-1] + 1)]
        match_out_table = [[] for i in range(sizes[-1] + 1)]

        # Nested loops through the sizes of two signatures, and the size of their overlap.
        for mi_size in sizes:

            for mo_size in sizes:

                for ov_size in range(1, 1 + (mi_size if mi_size < mo_size else
                                             mo_size)):

                    # Skip if the similarity size score doesn't meet the threshold.
                    similarity = self.similarity_size(mi_size, mo_size, ov_size)
                    if similarity < sim_threshold: continue

                    # Add [size,overlap] pair to MO table.
                    match_out_table[mo_size].append([mi_size, ov_size])

                    # Add overlap size to MI table. (But don't add duplicates.)
                    if not (mark_in_table[mi_size] and
                            any([x == ov_size
                                 for x in mark_in_table[mi_size]])):
                        mark_in_table[mi_size].append(ov_size)

                    break  # Only add the smallest matching overlap; skip the larger ones.

        return mark_in_table, match_out_table   # Return the two constructed tables.

    def cluster (self, sig_num, sim_threshold, sizes, elements):
        """Take an input data signature and assign it to a cluster.

        This is the core function of the MIMOSA implementation.

        Arguments:
        sig_num -- Index of the current input, in input series.
        sim_threshold -- Similarity threshold, between 0 and 1.
        sizes -- List of allowable sizes of signatures.
        elements -- The elements of the input signature."""

        sig_size     = len(elements)                # The signature size.
        combos_sig   = self.combinations[sig_size]  # Find the needed portions
        mo_sig_table = self.mo_table    [sig_size]  #   of precomputed tables
        mi_sig_table = self.mi_table    [sig_size]  #   for this sig size.
        markers      = self.markers                 # Storage for markers.
        cluster_id   = sig_num                      # Initialize the cluster assignment.

        # Build the partial sigs needed for this signature.
        partial_sigs = [[] for ov_size in range(mi_sig_table[-1] + 1)]
        for ov_size in mi_sig_table:

            # Map the precomputed index values for each combination ...
            for combo in combos_sig[ov_size]:

                # ... into the actual elements of this signature.
                partial_sigs[ov_size].append(
                    "-".join([elements[i] for i in combo]))

        # Match-Out stage.
        # For each possible size of another signature that could be similar to the signature ...
        for row in mo_sig_table:

            mi_size, ov_size = list(row)

            # ... check whether any of the partial signatures for that size is marked.
            for psig in partial_sigs[ov_size]:

                # Construct an MO key by concatenating a size value with the partial
                #   signature, and check whether the key is in the hash table.
                mo_token          = str(mi_size) + "-" + psig
                marker_cluster_id = markers.get(mo_token)
                if not marker_cluster_id: continue

                # If more than one MO key is found, assign the one that is
                #   marked with the earliest-numbered cluster.
                if marker_cluster_id < cluster_id:
                    cluster_id = marker_cluster_id

        # Mark-In stage.
        if cluster_id == sig_num:  # If an existing cluster was not found ...

            # ... construct MI keys for each possible overlap size for another
            #   signature that could be similar to the signature.
            for ov_size in mi_sig_table:

                # Construct an MI key by concatenating the signature size with a
                #   partial signature.  Mark all the MI keys that were not marked before.
                for ps in partial_sigs[ov_size]:
                    mi_token = str(sig_size) + "-" + ps
                    markers.setdefault(mi_token, cluster_id)

        # Return the selected cluster ID.
        return cluster_id


class DataClusterLinearJaccard (DataClusterLinear):
    """Implement the Jaccard similarity size function.

    A similarity SIZE function requires only set sizes and the
    overlap size, not the sets themselves."""

    def __init__ (self, args):
        """Pass args up to super class initializer."""
        super().__init__(args)

    def similarity_size (self, n_mark_in, n_match_out, n_overlap):
        """Jaccard: Divide size of intersection by size of union.

        Arguments:
        mi_size -- Number of elements in one of the sets.
        mo_size -- Number of elements in the other set.
        ov_size -- Number of elements in the overlap (intersection) of
                   the two sets."""
        return n_overlap / (n_mark_in + n_match_out - n_overlap)  # Jaccard.


class DataClusterCentroid (DataCluster, abc.ABC):
    """Implement a simple centroid clustering algorithm.

    The first assigned signature in a cluster is designated the centroid.
    Subsequent signatures matching the centroid are assigned to the
        cluster, but do not enlarge the cluster neighborhood.
    When an input signature matches more than one centroid, the input is
        assigned to the earliest (lowest numbered) corresponding cluster."""

    @abc.abstractmethod
    def similarity (self, setA, setB):
        """The similarity() method must be implemented by a subclass.

        Arguments:
        setA -- One of the sets.
        setB -- The other set."""
        pass

    def __init__ (self, args):
        """Copy arg key-values as self attrs.

        Arguments:
        args -- A dictionary whose keys are to become attributes of
                the DataClusterLinear object.  Required keys are:
                sim_threshold, sizes."""
        if not args.get('sizes'): raise ValueError('Sizes must be provided')
        for key in args.keys(): setattr(self, key, args[key])
        self.clusters = []

    def cluster (self, sig_num, sim_threshold, sizes, elements):
        """Take an input signature and assign it to a cluster.

        Arguments:
        sig_num -- Index of the current input, in input series.
        sim_threshold -- Similarity threshold, between 0 and 1.
        sizes -- List of allowable sizes of signatures.
        elements -- The elements of the input signature."""
        clusters   = self.clusters  # List of existing clusters.
        cluster_id = sig_num        # Initialize the cluster assignment.

        # Sequentially check all existing cluster centroids.
        for cluster in clusters:

            c_id, c_elements = list(cluster)   # The cluster ID and centroid elements.

            # Invoke the similarity function to compare the input signature elements
            #   to the existing centroid elements.  Skip if not similar.
            if self.similarity(elements, c_elements) < sim_threshold: continue

            cluster_id = c_id  # Assign the cluster ID of the first similar centroid.
            break              # And stop checking further centroids.

        # If a new cluster ID was assigned, record its centroid.
        if cluster_id == sig_num: clusters.append([cluster_id, elements])

        return cluster_id      # Return the assigned cluster ID.


class DataClusterCentroidJaccard (DataClusterCentroid):
    """Implement a Jaccard similarity measure for DataClusterCentroid."""

    def __init__ (self, args):
        """Pass args up to super class initializer."""
        super().__init__(args)

    #
    # This subclass of DataClusterCentroid implements a Jaccard similarity measure.
    #
    def similarity (self, A, B):
        """Jaccard: Divide size of intersection by size of union.

        Arguments:
        A -- One of the sets. (Can be a list or set.)
        B -- The other set. (Can be a list or set.)"""
        A = set(A)  # Convert to set (get rid of duplicates, etc.).
        B = set(B)
        sUnion = len(A.union(B))
        if sUnion == 0: return 0                 # Prevent division by zero.
        return len(A.intersection(B)) / sUnion   # Intersection size / union size.


def read_command_line ():
    """Read and validate the command-line arguments."""
    run           = ''
    sim_threshold = 0
    sizes         = []
    cl_args       = iter(sys.argv[1:])

    # Loop through each argument.
    for arg in cl_args:

        # Help requested.
        if   arg == '-h': usage()

        # Centroid (-c) or MIMOSA (-m).
        elif arg == '-c' or arg == '-m':
            if run: usage("Too many -c or -m args")
            run     = arg
            n_items = int(valid_next_arg(next(cl_args),
                                         arg,
                                         r'^\d+$'))

        # List of allowed signature size values (number of elements per signature).
        elif arg == '-s':
            if sizes: usage("Too many -s args")
            arg    = valid_next_arg(next(cl_args),
                                    arg,
                                    r'^\d+((,|\-|\.\.)\d+)*$')
            p      = re.compile(r'(\d+)')
            sizes  = []
            for range_term in arg.split(sep=","):
                nums   = p.findall(range_term)
                nfirst = int(nums[ 0])
                nlast  = int(nums[-1])
                sizes.extend(range(nfirst, nlast + 1))
            sizes  = sorted(set(sizes))  # Uniq.

        # Similarity threshold value.
        elif arg == '-t':
            if sim_threshold: usage("Too many -t args")
            sim_threshold = float(valid_next_arg(next(cl_args),
                                                 arg,
                                                 r'^[\d\.]+$'))

        else: usage("Invalid arg: " + arg)

    # Final check for command-line errors.
    err = []
    if not run          : err.append("Missing -m or -c count")
    if not sim_threshold: err.append("Missing -t threshold")
    if not sizes        : err.append("Missing -s sizes list")
    if err              : usage(err)

    # Return the validated values from the command line.
    return run, n_items, sim_threshold, sizes


def valid_next_arg (arg, prev_arg, regex):
    """Check whether arg exists and satisfies the regex.

    Arguments:
    arg -- The current command-line argument value.
    prev_arg -- The previous command-line argument value.
    regex -- Regular expression to validate arg."""
    if not arg: usage("Missing value after " + prev_arg)
    p = re.compile(regex)
    if not p.match(arg): usage("Invalid value after " + prev_arg + " : " + arg)
    return arg


def usage (messages = []):
    """Print out error messages (if any), and usage instructions.

    Arguments:
    messages -- List of error messages (optional)."""
    for message in messages: print(message, flush=True)
    str = """
USAGE:
    {PROGNAME} ARGS

    Required ARGS:
      -t N      Similarity threshold, 0 < N < 1.
      -s NLIST  Signature sizes list (numbers with "," and "-").

      -c N      Centroid run, N data items.
      OR
      -m N      MIMOSA run, N data items.

    Example:
      {PROGNAME} -t 0.6 -s 2-10 -m 10000000 < data.txt
"""
    print(str.format(PROGNAME=sys.argv[0]), flush=True)
    sys.exit(1 if messages else 0)


def main ():
    """Run the benchmark.

    This program is a wrapper for the two clustering algorithms.
    It is invoked from the command line, initializes one of the
    algorithms, runs the benchmark, and records the output."""

    sys.stdout.flush()
    run, n_items, sim_threshold, sizes = read_command_line()  # Get instructions.

    sig_num  = 1                                              # Data item number.
    max_size = sizes[-1]                                      # Max signature size.
    args     = { 'n_items'      : n_items      ,              # Bundle the args.
                 'sim_threshold': sim_threshold,
                 'sizes'        : sizes        , }
    clust    = (DataClusterLinearJaccard  (args) if run == '-m' else
                DataClusterCentroidJaccard(args))
    t0       = time.time()

    # Read data item signatures from STDIN.
    for line in sys.stdin:

        line       = line.strip()                             # Remove newline.
        elements   = line.split(sep="-")                      # Split line at hyphens.
        if not elements: continue                             # Skip empty lines.
        elements   = elements[:max_size]                      # Chop off excess terms.

        # Perform clustering on the signature.  Each algorithm must have a cluster() function.
        cluster_id = clust.cluster(sig_num, sim_threshold, sizes, elements)

        # Record the assigned cluster ID, the input number, and the signature elements.
        print("%8d %8d %s" % (cluster_id, sig_num, " ".join(elements)),
              flush=True)

        # Record an elapsed timestamp after every 1000 signatures
        #   (every 10 signatures during the first 2000).
        if sig_num <= 2000 and not (sig_num % 10) or not (sig_num % 1000):
            print("%d ----- %.6f" % (sig_num, time.time() - t0), flush=True)

        sig_num = sig_num + 1
        if sig_num > n_items: break   # Exit when requested number is done.

    # Report memory usage.
    print("Memory usage: " +
          str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
          flush=True)
    #    ---- Exit.

if __name__ == '__main__': main()

