"""Exactly the archived beam decisions, with incremental CX scoring.

No heuristic, tie break, candidate pool, or circuit contract changes. For unique
ordered support tuples the new term costs 2(|support|-1) CX minus twice the
common prefix of control lists when adjacent terms share their parity target.
The intervening RZ prevents cancellation across any earlier boundary.
"""

def order_parity_terms(records,policy,beam_width=4,candidate_pool=10):
    if policy!='beam_search':raise ValueError(policy)
    if not records:return []
    mapping={int(r['term_id']):r for r in records}
    assert len(mapping)==len(records)
    supports={i:tuple(r['support']) for i,r in mapping.items()}
    assert all(s and len(s)==len(set(s)) for s in supports.values())
    bits={i:sum(1<<q for q in s) for i,s in supports.items()}
    def increment(left,right):
        a,b=supports[left],supports[right];common=0
        if a[-1]==b[-1]:
            for x,y in zip(a[:-1],b[:-1]):
                if x!=y:break
                common+=1
        return 2*(len(b)-1-common)
    cache={}
    def candidates(last,remaining):
        if last not in cache:
            cache[last]=sorted(mapping,key=lambda i:(-(bits[last]&bits[i]).bit_count(),(bits[last]^bits[i]).bit_count(),mapping[i]['support_size'],mapping[i]['mask'],mapping[i]['term_id']))
        result=[]
        for i in cache[last]:
            if i in remaining:
                result.append(i)
                if len(result)==candidate_pool:break
        return result
    starts=sorted(records,key=lambda r:(r['support_size'],r['mask'],r['term_id']))[:beam_width]
    beam=[([int(r['term_id'])],set(mapping)-{int(r['term_id'])},2*(len(r['support'])-1)) for r in starts]
    while beam[0][1]:
        following=[]
        for prefix,remaining,score in beam:
            for i in candidates(prefix[-1],remaining):
                following.append((prefix+[i],remaining-{i},score+increment(prefix[-1],i)))
        following.sort(key=lambda state:state[2])  # Stable, exactly like the archive.
        beam=following[:beam_width]
    best=min(beam,key=lambda state:state[2])
    return [mapping[i] for i in best[0]]
