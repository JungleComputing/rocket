package nl.esciencecenter.rocket.util;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

public class Future<T> {
    private CompletableFuture<T> fut;

    public Future() {
        this(new CompletableFuture<>());
    }

    private Future(CompletableFuture<T> fut) {
        this.fut = fut;
    }

    static public <T> Future<T> ready(T value) {
        return Future.of(CompletableFuture.completedFuture(value));
    }

    static public <T> Future<T> exceptionally(Throwable value) {
        CompletableFuture<T> f = new CompletableFuture<>();
        f.completeExceptionally(value);
        return Future.of(f);
    }

    static public Future<Void> readyVoid() {
        return ready(null);
    }

    static public <T> Future<T> run(Supplier<Future<T>> f) {
        try {
            return f.get();
        } catch (Throwable e) {
            return Future.exceptionally(e);
        }
    }

    static public <T> Future<T> runAsync(Executor e, Supplier<T> f) {
        return Future.of(CompletableFuture.supplyAsync(f, e));
    }

    static public <T> Future<T> of(CompletableFuture<? extends T> f) {
        return new Future(f);
    }

    static public Future<Void> all(Future<?> ... futures) {
        switch (futures.length) {
            case 0:
                return Future.readyVoid();

            case 1:
                return futures[0].andVoid();

            case 2:
                return Future.join(futures[0], futures[1], (a, b) -> null);

            default:
                CompletableFuture<?>[] cfutures = new CompletableFuture[futures.length];

                for (int i = 0; i < futures.length; i++) {
                    cfutures[i] = futures[i].fut;
                }

                return Future.of(CompletableFuture.allOf(cfutures));
        }
    }

    static public <T> Future<Void> all(List<Future<T>> futures) {
        return Future.all(futures.toArray(new Future[0]));
    }

    static public <T, A, B> Future<T> join(Future<A> a, Future<B> b, BiFunction<? super A, ? super B, ? extends T> f) {
        return Future.of(a.fut.thenCombine(b.fut, f));
    }

    static public <A, B> Future<Tuple<A, B>> join(Future<A> a, Future<B> b) {
        return join(a, b, Tuple::new);
    }

    static public <T, A, B> Future<T> joinCompose(Future<A> a, Future<B> b, BiFunction<? super A, ? super B, Future<T>> f) {
        return Future.of(a.fut.thenCombine(b.fut, f)).then(x -> x);
    }

    public Future<Void> andVoid() {
        return Future.of(fut.thenRun(() -> {}));
    }

    public <R> Future<R> andReturn(R value) {
        return Future.of(fut.thenApply(__ -> value));
    }

    public <R> Future<R> then(Function<? super T, Future<R>> f) {
        return Future.of(fut.thenCompose(x -> f.apply(x).fut));
    }

    public <B, R> Future<R> thenJoin(Future<B> b, BiFunction<? super T, ? super B, ? extends R> f) {
        return Future.join(this, b, f);
    }

    public <R> Future<R> thenAsync(Executor e, Function<? super T, Future<R>> f) {
        return Future.of(fut.thenComposeAsync(x -> f.apply(x).fut, e));
    }

    public <R> Future<R> thenMap(Function<? super T, ? extends R> f) {
        return Future.of(fut.thenApply(f));
    }

    public <R> Future<R> thenMapAsync(Executor e, Function<? super T, ? extends R> f) {
        return Future.of(fut.thenApplyAsync(f, e));
    }

    public Future<T> thenRun(Consumer<T> f) {
        return thenMap(arg -> {
            f.accept(arg);
            return arg;
        });
    }

    public Future<T> thenRunAsync(Executor e, Consumer<T> f) {
        return thenMapAsync(e, arg -> {
            f.accept(arg);
            return arg;
        });
    }

    public Future<T> onComplete(BiConsumer<? super T, Throwable> f) {
        return Future.of(fut.whenComplete(f));
    }

    public Future<T> handleException(Function<Throwable, ? extends T> f) {
        return Future.of(fut.handle((r, e) -> {
            if (e != null) {
                return f.apply(e);
            } else {
                return r;
            }
        }));
    }

    public boolean complete(T value) {
        return fut.complete(value);
    }

    public Future<T> thenComplete(Future<T> f) {
        return onComplete((result, error) -> {
            if (error == null) {
                f.fut.complete(result);
            } else {
                f.fut.completeExceptionally(error);
            }
        });
    }

    public boolean completeVoid() {
        return complete((T) null);
    }

    public boolean completeExceptionally(Throwable value) {
        return fut.completeExceptionally(value);
    }

    public T get() {
        return fut.getNow(null);
    }

    public T await() throws InterruptedException, ExecutionException {
        return fut.get();
    }

    public T await(long l, TimeUnit u) throws InterruptedException, ExecutionException, TimeoutException {
        return fut.get(l, u);
    }

    public boolean isCompleted() {
        return fut.isDone();
    }

    public boolean isCompletedSuccessfully() {
        return fut.isDone() && !fut.isCompletedExceptionally();
    }

    public boolean isCompletedExceptionally() {
        return fut.isCompletedExceptionally();
    }

    public void completeAsync(Executor exec, T value) {
        fut.completeAsync(() -> value, exec);
    }

    public void completeVoidAsync(Executor exec) {
        completeAsync(exec, null);
    }

    public void completeExceptionallyAsync(Executor exec, Throwable value) {
        fut.completeExceptionally(value);
    }

    public CompletableFuture<T> getCompletableFuture() {
        return fut;
    }
}
