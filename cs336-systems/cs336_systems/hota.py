from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir = "./trace")
overlap_df = analyzer.get_comm_comp_overlap(False)
print(overlap_df)