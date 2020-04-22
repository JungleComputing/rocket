package nl.esciencecenter.rocket.indexspace;

public interface IndexSpace<K> {
    public void divide(IndexSpaceDivision<K> ctx);
    int size();
}
