package nl.esciencecenter.rocket.util;

import nl.esciencecenter.xenon.XenonException;
import nl.esciencecenter.xenon.credentials.PasswordCredential;
import nl.esciencecenter.xenon.filesystems.FileSystem;
import nl.esciencecenter.xenon.filesystems.Path;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;

public class FileSystemFactory {
    public static FileSystem create(String key) throws XenonException {
        String[] parts = key.split(":", 2);

        if (parts.length == 1) {
            return FileSystem.create("file", key);
        }

        if (parts[0].equals("s3")) {
            String[] p = parts[1].split(":", 2);
            String[] q = (p.length > 1 ? p[1] : "").split("/", 2);

            return createS3Adapter(
                    q[0],
                    q[1],
                    p[0]);
        }

        if (parts[0].equals("file")) {
            return FileSystem.create("file", parts[1]);
        }

        if (parts[0].equals("sftp")) {
            return FileSystem.create("sftp", parts[1]);
        }

        throw new IllegalArgumentException("the string '" + key + "' is not a valid location identifier");
    }

    private static FileSystem createS3Adapter(String hostName, String location, String credentialsFile) throws XenonException {
        String url, accessKey, secretKey;

        try (InputStream reader = Files.newInputStream(Paths.get(credentialsFile))) {
            JSONObject obj = new JSONObject(new JSONTokener(reader));
            JSONObject hosts = obj.optJSONObject("hosts");

            if (hosts == null) {
                throw new JSONException("could not find key 'hosts' in " + credentialsFile);
            }

            JSONObject host = hosts.optJSONObject(hostName);
            if (host == null) {
                throw new JSONException("could not find key '" + hostName + "' in " + credentialsFile);
            }

            url = host.optString("url");
            accessKey = host.optString("accessKey");
            secretKey = host.optString("secretKey");
        } catch (Exception e) {
            throw new XenonException("s3", "failed to parse configuration file", e);
        }


        return FileSystem.create(
                "s3",
                url + "/" + location,
                new PasswordCredential(accessKey, secretKey));
    }

    public static Path resolvePath(FileSystem fs, Path p) throws XenonException {
        return fs.getAttributes(p).isSymbolicLink() ?
                resolvePath(fs, fs.readSymbolicLink(p)) :
                p;
    }
}
