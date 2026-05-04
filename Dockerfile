FROM axolotlai/axolotl:main-latest

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh
WORKDIR /app
CMD ["/app/run.sh"]
