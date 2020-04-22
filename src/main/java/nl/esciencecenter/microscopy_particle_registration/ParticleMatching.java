package nl.esciencecenter.microscopy_particle_registration;

import java.io.Serializable;

public class ParticleMatching implements Serializable {
    public double score;
    public double translate_x;
    public double translate_y;
    public double rotation;

    public ParticleMatching(double score, double translate_x, double translate_y, double rotation) {
        this.score = score;
        this.translate_x = translate_x;
        this.translate_y = translate_y;
        this.rotation = rotation;
    }
}
